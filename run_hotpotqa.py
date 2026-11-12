"""Sequential online benchmark. Only corpus and question enter retrieval/learning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from main import LLM, answer_question
from retrieve import GraphRetriever
from evaluate import evaluate_answers


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='dataset/hotpotqa.json')
    parser.add_argument('--corpus',default='dataset/hotpotqa_corpus.json')
    parser.add_argument('--run-dir',default='runs/hotpotqa_two_edge_types')
    parser.add_argument('--encoder',default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--model',default=os.getenv('DEEPSEEK_MODEL','deepseek-flash'))
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--explicit-degree', type=int, default=8,
                        help='Maximum explicit neighbors per paragraph (0 disables explicit edges)')
    parser.add_argument('--explicit-weight', type=float, default=0.25)
    parser.add_argument('--global-every', type=int, default=100)
    args=parser.parse_args()
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=True)
    rows=json.loads(Path(args.dataset).read_text())
    if args.limit: rows=rows[:args.limit]
    paragraphs=[dict(id=str(p['idx']),text=f"{p['title']}\n{p['text']}") for p in json.loads(Path(args.corpus).read_text())]
    config=vars(args)|{'dataset_sha256':hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                        'corpus_sha256':hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()}
    config_path=root/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Run configuration changed; choose a new run directory')
    write_json(config_path,config)
    llm=LLM(args.model)
    # Probe auth/network/model before downloading encoder or starting benchmark.
    probe=llm('Reply only OK.')
    print('API preflight successful',flush=True)
    retriever=GraphRetriever(paragraphs,str(root/'state'),args.encoder,
                             explicit_degree=args.explicit_degree, explicit_weight=args.explicit_weight)
    print(f'Graph initialized with {len(retriever.explicit_edges)} explicit and '
          f'{len(retriever.implicit_edges)} implicit edges', flush=True)
    records=[]
    for index,row in enumerate(rows):
        path=root/f'{index:04d}.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['_id'] != row['_id']: raise ValueError('Question order changed')
        else:
            # Stop rather than double-train if interrupted between model update and trace save.
            if retriever.questions != index:
                raise RuntimeError('State/trace mismatch: restore matching checkpoint before resuming')
            result=answer_question(row['question'],retriever,llm,top_k=args.top_k,
                                   global_every=args.global_every)
            result.update(_id=row['_id'],correct_answer=row['answer'])
            write_json(path,result)
        if 'evaluation' not in result:
            # Gold answer is accessed only after graph learning, solely for evaluation.
            result['evaluation']=evaluate_answers([row['question']],[result['answer']],[row['answer']],mode='both',llm=llm)
            write_json(path,result)
        records.append(result)
        n=len(records)
        summary={'completed':n,'total':len(rows),'model':args.model,
                 'ContainsMatch':sum(r['evaluation']['scores']['ContainsMatch'] for r in records)/n,
                 'LLMAccuracy':sum(r['evaluation']['scores']['LLMAccuracy'] for r in records)/n}
        write_json(root/'summary.json',summary)
        print(json.dumps(summary),flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
