import json
import tempfile
import unittest
import torch
from retrieve import GraphRetriever, distance
from main import answer_question, decompose_question, judge_links


class Encoder:
    def encode(self,texts):
        values = {'A':[1.,0.,0.], 'B':[0.,1.,0.], 'C':[0.,0.,1.], 'D':[-1.,0.,0.]}
        return [values.get(t,[1.,0.,0.]) for t in texts]


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.corpus = [dict(id=t,text=t) for t in 'ABCD']
        self.r = GraphRetriever(self.corpus,self.tmp.name,encoder=Encoder())

    def test_update_and_reload(self):
        before = self.r.points.clone()
        d = distance(before[0],before[1])
        self.r.learn('bridge', [dict(source='A',target='B',reason='complementary evidence')])
        self.assertLess(distance(self.r.points[0],self.r.points[1]),d)
        self.assertTrue(torch.equal(before[2:],self.r.points[2:]))
        self.assertLess((self.r.points-before).norm(dim=-1).max(), 0.001)
        self.assertTrue(torch.equal(self.r.semantic, self.r.base/0.2))
        reloaded = GraphRetriever(self.corpus,self.tmp.name,encoder=Encoder())
        self.assertTrue(torch.equal(reloaded.points,self.r.points))
        self.assertEqual(reloaded.questions,1)
        self.assertEqual(reloaded.edges[0]['question'],'bridge')
        self.assertTrue(torch.isfinite(self.r.points).all())
        self.assertTrue((self.r.points.norm(dim=-1)<1).all())

    def test_dual_recall(self):
        self.assertEqual([x['id'] for x in self.r.retrieve('A',1)],['A'])
        self.r.learn('bridge',[dict(source='A',target='B',reason='bridge')])
        hits = self.r.retrieve('A',1)
        self.assertEqual([x['id'] for x in hits],['A','B'])
        self.assertIn('hyperbolic',hits[1]['sources'])

    def test_global_and_no_links(self):
        self.r.learn('q',[dict(source='A',target='B',reason='bridge')],every=1)
        self.r.learn('q2',[],every=1)
        self.assertEqual(self.r.questions,2)
        self.assertTrue(torch.isfinite(self.r.points).all())

    def test_pipeline_and_validation(self):
        prompts=[]
        def llm(prompt,json_mode=False):
            prompts.append(prompt)
            if json_mode:
                return {'links':[dict(source='A',target='B',reason='bridge'),
                                 dict(source='A',target='missing',reason='invalid')]}
            if 'The given question Q:' in prompt:
                return "['1.First?', '2.Second?']"
            return 'answer'
        result = answer_question('original',self.r,llm,top_k=2)
        self.assertEqual(len(result['steps']),2)
        self.assertEqual(len(result['links']),1)
        self.assertEqual(result['questions_seen'],1)
        self.assertEqual(decompose_question('simple',lambda p:'End.'),['simple'])
        self.assertEqual(decompose_question('q',lambda p:'1. One?\n2. Two?'),['One?','Two?'])
        self.assertIn('Previous answers: ["answer"]',prompts[2])
        with self.assertRaises(ValueError):
            decompose_question('q',lambda p:'nonsense')

    def test_one_epoch_covers_all_edges(self):
        from unittest.mock import patch
        import geoopt
        # 257 observations must produce two batches, exactly once each.
        links = [dict(source='A',target='B',reason='bridge') for _ in range(257)]
        original = geoopt.optim.RiemannianAdam.step
        calls = []
        def step(optimizer, *args, **kwargs):
            calls.append(1)
            return original(optimizer, *args, **kwargs)
        with patch.object(geoopt.optim.RiemannianAdam, 'step', step):
            self.r.learn('q',links,every=0)
        self.assertEqual(len(calls),2)

    def test_cumulative_drift_bound(self):
        self.r.max_drift = 0.0002
        original = self.r.semantic.clone()
        for _ in range(12):
            self.r.learn('q',[dict(source='A',target='B',reason='bridge')],every=0)
        drift = (self.r.points-self.r.base).norm(dim=-1)
        self.assertLessEqual(float(drift.max()),self.r.max_drift+1e-7)
        self.assertTrue(torch.equal(original,self.r.semantic))
        saved = torch.load(self.r.path,weights_only=True)
        self.assertEqual(saved['training']['epochs'],1)
        self.assertEqual(saved['training']['learning_rate'],1e-4)

    def test_corpus_change_rejected(self):
        self.r.save()
        with self.assertRaises(ValueError):
            GraphRetriever([dict(id='different',text='A')],self.tmp.name,encoder=Encoder())


if __name__ == '__main__':
    unittest.main()
