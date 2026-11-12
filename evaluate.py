"""Answer evaluation, independent of paragraph-link judgment and graph training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import string
from pathlib import Path


def normalize_answer(answer: str) -> str:
    text = ''.join(ch for ch in answer.lower() if ch not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def calculate_metric_scores_contains(gold_answers, predicted_answers):
    if len(gold_answers) != len(predicted_answers):
        raise ValueError('Reference and predicted answers must have the same length.')
    results = []
    for gold, predicted in zip(gold_answers, predicted_answers):
        gold_norm, pred_norm = normalize_answer(gold), normalize_answer(predicted)
        # An empty gold must not match every prediction through Python substring semantics.
        results.append({'ContainsMatch': int(bool(gold_norm) and gold_norm in pred_norm)})
    return {'ContainsMatch': sum(r['ContainsMatch'] for r in results)/len(results) if results else 0}, results


def calculate_yes_ratio(results):
    return sum(r.strip().lower() == 'yes' for r in results)/len(results) if results else 0


def evaluate_multiple(questions, generated_answers, correct_answers, llm, max_workers=5):
    if not len(questions) == len(generated_answers) == len(correct_answers):
        raise ValueError('Questions, generated answers, and reference answers must have the same length.')
    templates = [
        'Evaluate the generated answer against the correct answer. Only answer yes if the generated answer is correct, no otherwise. '
        f'question: {q}\ngenerated_answer: {generated}\ncorrect_answer: {correct}\nevaluation: '
        for q, generated, correct in zip(questions, generated_answers, correct_answers)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(llm, templates))
    # Failures propagate; never score an API error as a wrong answer silently.
    return results, calculate_yes_ratio(results)


def evaluate_answers(questions, generated_answers, correct_answers, mode='both', llm=None):
    if mode not in ('str-match', 'llm', 'both'):
        raise ValueError('Unknown evaluation mode')
    if not len(questions) == len(generated_answers) == len(correct_answers):
        raise ValueError('Input lengths differ')
    scores, examples = {}, [{} for _ in questions]
    if mode in ('str-match', 'both'):
        pooled, matches = calculate_metric_scores_contains(correct_answers, generated_answers)
        scores.update(pooled)
        for row, match in zip(examples, matches):
            row.update(match)
    if mode in ('llm', 'both'):
        if llm is None:
            raise ValueError('LLM evaluator required')
        results, acc = evaluate_multiple(questions, generated_answers, correct_answers, llm)
        scores['LLMAccuracy'] = acc
        for row, result in zip(examples, results):
            row['LLMJudge'] = result
    return {'scores': scores, 'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='JSONL with question, answer, correct_answer')
    parser.add_argument('--mode', choices=['str-match','llm','both'], default='str-match')
    parser.add_argument('--model', default=None)
    parser.add_argument('--output', default='evaluation.json')
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    llm = None
    if args.mode != 'str-match':
        from main import LLM
        import os
        llm = LLM(args.model or os.getenv('DEEPSEEK_MODEL','deepseek-flash'))
    result = evaluate_answers([r['question'] for r in rows], [r['answer'] for r in rows],
                              [r['correct_answer'] for r in rows], args.mode, llm)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result['scores']))


if __name__ == '__main__':
    main()
