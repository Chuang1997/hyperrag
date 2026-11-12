"""Frozen SBERT semantics + online Poincare paragraph coordinates (curvature -1)."""
import hashlib
import json
from pathlib import Path
import torch
import geoopt
import torch.nn.functional as F
from topic_graph import build_explicit_edges


BALL = geoopt.PoincareBall(c=1.0)


def distance(x, y):
    return BALL.dist(x, y)


class GraphRetriever:
    def __init__(self, paragraphs, state_dir='state', model_name='sentence-transformers/all-MiniLM-L6-v2', encoder=None, learning_rate=1e-4, anchor_weight=10.0, max_drift=0.01,
                 explicit_degree=8, explicit_weight=0.25):
        if learning_rate <= 0 or anchor_weight < 0 or max_drift <= 0:
            raise ValueError('Require learning_rate > 0, anchor_weight >= 0, max_drift > 0')
        if explicit_degree < 0 or explicit_weight <= 0:
            raise ValueError('Require explicit_degree >= 0 and explicit_weight > 0')
        self.explicit_weight = explicit_weight
        self.learning_rate = learning_rate
        self.anchor_weight = anchor_weight
        self.max_drift = max_drift
        self.paragraphs = paragraphs
        self.ids = [p['id'] for p in paragraphs]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('Corpus must be nonempty with unique paragraph IDs')
        self.lookup = {pid:i for i,pid in enumerate(self.ids)}
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(model_name)
        self.encoder = encoder
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'index.pt'
        signature = hashlib.sha256(json.dumps(['two-edge-types-v1', model_name, paragraphs, explicit_degree, explicit_weight], sort_keys=True).encode()).hexdigest()
        self.implicit_edges, self.questions = [], 0
        restored = self.path.exists()
        if restored:
            saved = torch.load(self.path, map_location='cpu', weights_only=True)
            if saved['signature'] != signature:
                raise ValueError('Corpus/model changed: use a new state directory')
            self.semantic, self.points = saved['semantic'], saved['points']
            self.explicit_edges = saved['explicit_edges']
            self.implicit_edges = saved['implicit_edges']
            self.questions = saved['questions']
        else:
            self.explicit_edges = build_explicit_edges(paragraphs, explicit_degree)
            self.semantic = self.encode([p['text'] for p in paragraphs])
            self.points = self.semantic.clone() * 0.2  # exp_0(atanh(.2) * unit SBERT)
        self.base = self.semantic * 0.2
        # Also bound drift when resuming an older checkpoint.
        self.points = self.bound_drift(self.points, self.base)
        self.signature = signature
        if not restored:
            if self.explicit_edges:
                self.optimize(set(range(len(self.ids))))
            self.save()

    @property
    def edges(self):
        return self.explicit_edges + self.implicit_edges

    def encode(self, texts):
        return F.normalize(torch.as_tensor(self.encoder.encode(texts), dtype=torch.float32), dim=-1).cpu()

    def retrieve(self, query_text, top_k=3):
        """Keep SBERT top-k, union with top-k hyperbolic neighbors of SBERT anchors.

        Anchors are current-query dependent; learned coordinates need no query encoder retraining.
        """
        if top_k < 1:
            raise ValueError('top_k must be positive')
        k = min(top_k, len(self.ids))
        scores = (self.semantic @ self.encode([query_text])[0])
        semantic_ids = scores.topk(k).indices.tolist()
        hyper_ids = []
        if self.edges:
            # Exclude anchor itself: seek complementary evidence around semantic hits.
            d = distance(self.points[:,None,:], self.points[semantic_ids][None,:,:])
            for col, anchor in enumerate(semantic_ids):
                d[anchor,col] = float('inf')
            hyper_ids = d.min(dim=1).values.topk(k, largest=False).indices.tolist()
        merged = list(dict.fromkeys(semantic_ids + hyper_ids))
        return [dict(self.paragraphs[i], semantic_score=float(scores[i]),
                     sources=[name for name,indices in [('semantic',semantic_ids),('hyperbolic',hyper_ids)] if i in indices])
                for i in merged]

    def learn(self, question, links, every=100):
        """links already validated by caller; retain question + reason for auditing."""
        touched = set()
        for link in links:
            a,b = self.lookup[link['source']], self.lookup[link['target']]
            self.implicit_edges.append(dict(link, kind='implicit', question=question,
                                            question_number=self.questions+1))
            touched.update((a,b))
        self.questions += 1
        if touched:
            # Include one-hop historical neighbors; all other rows remain bitwise unchanged.
            neighbors = set(touched)
            for e in self.edges:
                a,b = self.lookup[e['source']], self.lookup[e['target']]
                if a in touched or b in touched:
                    neighbors.update((a,b))
            self.optimize(neighbors)
        if every > 0 and self.questions % every == 0 and self.edges:
            self.optimize(set(range(len(self.ids))))
        self.save()

    def bound_drift(self, points, base):
        # Hard cumulative Euclidean displacement cap from SBERT initialization.
        delta = points - base
        scale = (self.max_drift / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
        return BALL.projx(base + delta * scale)

    def optimize(self, active):
        indices = sorted(active)
        mapping = {v:i for i,v in enumerate(indices)}
        edges = self.edges
        selected = [e for e in edges if self.lookup[e['source']] in active and self.lookup[e['target']] in active]
        if not selected:
            return
        # Exactly one epoch: visit every selected edge once, in batches of 256.
        pairs = torch.tensor([[mapping[self.lookup[e['source']]], mapping[self.lookup[e['target']]]] for e in selected])
        known = {tuple(sorted((self.lookup[e['source']],self.lookup[e['target']]))) for e in edges}
        weights = torch.tensor([self.explicit_weight if e['kind'] == 'explicit' else 1.0
                                for e in selected])
        x = geoopt.ManifoldParameter(self.points[indices].clone(), manifold=BALL)
        optimizer = geoopt.optim.RiemannianAdam([x], lr=self.learning_rate, stabilize=1)
        order = torch.randperm(len(pairs))
        for batch_indices in order.split(256):
            optimizer.zero_grad()
            batch = pairs[batch_indices]
            a,b = batch[:,0],batch[:,1]
            pos = distance(x[a],x[b])
            # Never treat a known neighbor or the node itself as a negative.
            negative_pairs = []
            for batch_row, ai in enumerate(a.tolist()):
                for ni in torch.randint(len(self.ids),(8,)).tolist():
                    if ni != indices[ai] and tuple(sorted((indices[ai],ni))) not in known:
                        negative_pairs.append((batch_row, ai, ni))
                        break
            batch_weights = weights[batch_indices]
            loss = (batch_weights * pos.square()).mean()
            if negative_pairs:
                negative_rows, na, nb = zip(*negative_pairs)
                targets = torch.stack([x[mapping[n]] if n in mapping else self.points[n] for n in nb])
                penalty = F.relu(1.0 - distance(x[list(na)], targets)).square()
                loss = loss + (batch_weights[list(negative_rows)] * penalty).mean()
            loss = loss + self.anchor_weight * distance(x, self.base[indices]).square().mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.copy_(self.bound_drift(x, self.base[indices]))
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite hyperbolic update')
        self.points[indices] = x.detach()

    def save(self):
        tmp = self.path.with_suffix('.tmp')
        torch.save(dict(signature=self.signature, semantic=self.semantic, points=self.points,
                        explicit_edges=self.explicit_edges, implicit_edges=self.implicit_edges,
                        questions=self.questions,
                        training=dict(optimizer='geoopt.RiemannianAdam', epochs=1,
                                      learning_rate=self.learning_rate, anchor_weight=self.anchor_weight,
                                      max_drift=self.max_drift)), tmp)
        tmp.replace(self.path)
        for kind, edges in [('explicit', self.explicit_edges), ('implicit', self.implicit_edges)]:
            path = self.root / f'{kind}_edges.json'
            if kind == 'implicit' or not path.exists():
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(edges, ensure_ascii=False, indent=2))
                temporary.replace(path)
