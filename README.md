# HyperRAG

HyperRAG is a research prototype for multi-hop question answering. It combines Sentence-BERT retrieval with a paragraph graph learned from evidence that supports answers across questions. The graph updates paragraph embeddings in hyperbolic space so that later queries can retrieve evidence linked by reasoning as well as by semantic similarity.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
read -rs DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
python main.py --corpus corpus.example.jsonl --state state \
  --question "Who wrote Don Juan, and how was he related to Ada Lovelace?"
```

Paste your DeepSeek API key when the shell waits for input. The default model is `deepseek-flash`. Use `--model` or `DEEPSEEK_MODEL` to select a model and `DEEPSEEK_BASE_URL` to configure the endpoint.

The single-question CLI accepts a JSONL corpus with a unique string `id` and a nonempty `text` field per record. The default encoder is `sentence-transformers/all-MiniLM-L6-v2`. Use `--encoder` to load a local copy. Use a new state directory when changing the corpus or encoder.

## Retrieval and online learning

1. Decompose each question into ordered subquestions. Simple questions run as a single step.
2. Query Sentence-BERT with the current subquestion and, when available, the previous answer. Keep the top *k* paragraphs by cosine similarity.
3. Once the graph has edges, use the semantic hits as anchors and retrieve up to *k* additional candidates by minimum Poincaré distance to an anchor. Merge the two candidate lists and remove duplicates.
4. Answer each subquestion with the retrieved paragraphs and earlier answers, then synthesize the final answer.
5. Ask the LLM to identify paragraph pairs that jointly support the reasoning. Store validated edges with their originating question and rationale.
6. Update the linked paragraphs and their one-hop neighbors. Run a global update every 100 questions by default. New edges affect retrieval from the next question onward.

## Hyperbolic embeddings

Paragraph coordinates are initialized at 0.2 times the normalized Sentence-BERT embeddings on a Poincaré ball of curvature −1. Sentence-BERT embeddings stay fixed. Geoopt's Riemannian Adam optimizes the hyperbolic coordinates for one epoch per local or global update, with batches of up to 256 edges.

The objective pulls linked paragraphs together, pushes sampled negatives outside a distance margin, and penalizes drift from the initial coordinates. The default learning rate is `1e-4`, the anchor weight is `10`, and the maximum Euclidean displacement from initialization is `0.01`. Adjust these settings with `--learning-rate`, `--anchor-weight`, and `--max-drift` in the single-question CLI.

Retrieval uses exact distances over the corpus. Run questions sequentially when sharing a graph checkpoint.

## HotpotQA benchmark

```bash
python run_hotpotqa.py \
  --dataset dataset/hotpotqa.json \
  --corpus dataset/hotpotqa_corpus.json \
  --run-dir runs/hotpotqa_deepseek \
  --model deepseek-flash
```

The runner encodes paragraph titles and text and processes questions in dataset order. Retrieval and graph updates use the corpus and question text. Reference answers are used for scoring. The default retrieval budget is five semantic candidates and up to five hyperbolic candidates per subquestion.

Each completed question has a JSON record with its answer, retrieval trace, graph edges, and scores. `summary.json` reports aggregate scores for the completed questions. To resume a run, reuse its original configuration and checkpoint.

The [archived run](runs/hotpotqa_deepseek/summary.json) completed 190 of 1,000 questions before an API timeout. On those questions, ContainsMatch was **67.37%** and LLMAccuracy was **83.16%**.

## Answer evaluation

`evaluate.py` accepts JSONL records with `question`, `answer`, and `correct_answer` fields.

- **ContainsMatch** checks whether the normalized prediction contains the normalized reference. Normalization lowercases text, removes punctuation and English articles, and collapses whitespace.
- **LLMAccuracy** is the fraction of answers that the judge marks correct against the reference using a yes/no response.

```bash
python evaluate.py --input predictions.jsonl --mode str-match --output evaluation.json
python evaluate.py --input predictions.jsonl --mode both --model deepseek-flash --output evaluation.json
```

## Tests

The tests use a mock encoder and LLM and run without API credentials or model downloads.

```bash
python -m unittest discover -v
```
