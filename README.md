# HyperRAG

Code accompanying [Query-Aware Knowledge Retrieval via Hyperbolic Structuring](https://aclanthology.org/2026.acl-long.986.pdf) (ACL 2026).

## Hyperbolic graph construction

HyperRAG builds a graph over corpus passages. Shared keywords and phrases form explicit edges before any questions are processed. As each question is answered, an LLM identifies passages that jointly support its reasoning and adds query-aware implicit edges.

The graph is embedded in a Poincaré ball. The explicit edges shape the initial geometry. New implicit edges trigger local updates around the linked passages, and both edge types are used in periodic global updates.

## Run on HotpotQA

Install the dependencies with `pip install -r requirements.txt` and set `DEEPSEEK_API_KEY` in your environment.

```bash
python run_hotpotqa.py --limit 200 --run-dir runs/hotpotqa_new_run
```
