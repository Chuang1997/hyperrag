"""Build sparse explicit edges from shared corpus keywords and phrases."""
from collections import defaultdict
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_explicit_edges(paragraphs, max_degree=8):
    """Score shared top TF-IDF terms and keep a bounded undirected graph."""
    if max_degree == 0 or len(paragraphs) < 2:
        return []

    # Terms found in only one paragraph cannot define a shared topic.
    vectorizer = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2), min_df=2, max_df=64,
        strip_accents='unicode',
    )
    try:
        matrix = vectorizer.fit_transform(p['text'] for p in paragraphs)
    except ValueError as exc:
        if 'empty vocabulary' in str(exc) or 'no terms remain' in str(exc).lower():
            return []
        raise

    terms = vectorizer.get_feature_names_out()
    postings = defaultdict(list)
    for i, row in enumerate(matrix):
        ranked = sorted(zip(row.indices, row.data), key=lambda pair: (-pair[1], pair[0]))
        for term, weight in ranked[:8]:
            postings[term].append((i, float(weight)))

    scores = defaultdict(float)
    shared = defaultdict(list)
    for term, hits in postings.items():
        for (left, a), (right, b) in combinations(hits, 2):
            pair = left, right
            scores[pair] += a * b
            shared[pair].append(str(terms[term]))

    degrees = [0] * len(paragraphs)
    edges = []
    for left, right in sorted(scores, key=lambda pair: (-scores[pair], pair)):
        if max(degrees[left], degrees[right]) >= max_degree:
            continue
        edges.append({
            'source': paragraphs[left]['id'],
            'target': paragraphs[right]['id'],
            'kind': 'explicit',
            'keywords': sorted(shared[left, right]),
            'topic_score': scores[left, right],
        })
        degrees[left] += 1
        degrees[right] += 1
    return edges
