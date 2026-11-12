import unittest
from evaluate import normalize_answer, calculate_metric_scores_contains, evaluate_answers, evaluate_multiple


class Tests(unittest.TestCase):
    def test_contains_is_not_exact_match(self):
        self.assertEqual(normalize_answer('The Lord Byron!'), 'lord byron')
        pooled, rows = calculate_metric_scores_contains(['Byron','Ada',''],['Lord Byron wrote it','Babbage','anything'])
        self.assertEqual(pooled['ContainsMatch'],1/3)
        self.assertEqual([r['ContainsMatch'] for r in rows],[1,0,0])

    def test_two_modes_and_order(self):
        def judge(prompt):
            return ' YES ' if 'correct_answer: Byron' in prompt else 'no'
        result = evaluate_answers(['q1','q2'],['Lord Byron','wrong'],['Byron','Ada'],'both',judge)
        self.assertEqual(result['scores'],{'ContainsMatch':0.5,'LLMAccuracy':0.5})
        self.assertEqual(result['examples'][0]['LLMJudge'],' YES ')
        self.assertEqual(evaluate_answers([],[],[],mode="str-match")['scores'],{'ContainsMatch':0})

    def test_mismatch_and_api_failure(self):
        with self.assertRaises(ValueError):
            evaluate_multiple(['q'],[],[],lambda _: 'yes')
        def fail(prompt):
            raise RuntimeError('API failure')
        with self.assertRaises(RuntimeError):
            evaluate_multiple(['q'],['a'],['a'],fail)


if __name__ == '__main__':
    unittest.main()
