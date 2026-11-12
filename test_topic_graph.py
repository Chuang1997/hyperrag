import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import torch

from retrieve import GraphRetriever, distance
from topic_graph import build_explicit_edges


class Encoder:
    def encode(self, texts):
        return torch.eye(4)[:len(texts)]


class TopicGraphTests(unittest.TestCase):
    paragraphs = [
        {'id': 'a', 'text': 'Solar energy powers the rural grid.'},
        {'id': 'b', 'text': 'Solar energy supplies clean electricity.'},
        {'id': 'c', 'text': 'An unrelated violin performance.'},
        {'id': 'd', 'text': 'An unrelated violin concerto.'},
    ]

    def test_explicit_edges_are_sparse_and_auditable(self):
        edges = build_explicit_edges(self.paragraphs, max_degree=1)
        self.assertEqual(build_explicit_edges(self.paragraphs, 0), [])
        self.assertEqual(build_explicit_edges(self.paragraphs, 1), edges)
        self.assertEqual({(e['source'], e['target']) for e in edges}, {('a', 'b'), ('c', 'd')})
        self.assertIn('solar energy', edges[0]['keywords'] if edges[0]['source'] == 'a' else edges[1]['keywords'])
        degrees = Counter(node for e in edges for node in (e['source'], e['target']))
        self.assertLessEqual(max(degrees.values()), 1)

    def test_cold_start_online_updates_and_resume(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as root:
            retriever = GraphRetriever(self.paragraphs, root, encoder=Encoder(), explicit_degree=1)
            self.assertEqual(retriever.questions, 0)
            self.assertLess(distance(retriever.points[0], retriever.points[1]),
                            distance(retriever.base[0], retriever.base[1]))
            self.assertEqual(retriever.retrieve('solar', top_k=1)[1]['id'], 'b')
            with patch.object(retriever, 'optimize', wraps=retriever.optimize) as optimize:
                retriever.learn('How does solar power relate to the concert?', [
                    {'source': 'a', 'target': 'c', 'reason': 'Example evidence bridge'},
                ], every=1)
            self.assertEqual(optimize.call_count, 2)
            self.assertEqual(len(retriever.explicit_edges), 2)
            self.assertEqual(len(retriever.implicit_edges), 1)
            self.assertEqual(retriever.implicit_edges[0]['kind'], 'implicit')
            restored = GraphRetriever(self.paragraphs, root, encoder=Encoder(), explicit_degree=1)
            self.assertEqual(restored.questions, 1)
            self.assertEqual(restored.edges, retriever.edges)
            self.assertTrue(torch.equal(restored.points, retriever.points))
