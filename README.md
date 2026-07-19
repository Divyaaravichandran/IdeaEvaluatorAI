# Hackathon Idea Evaluation and Ranking System

This project evaluates and ranks hackathon ideas with a local ensemble of open-weight LLMs. It makes no calls to paid or hosted AI providers and requires no API keys.

## Default model ensemble

- Llama 3.2 (`llama3.2:3b`)
- Mistral (`mistral:7b`)
- Qwen 2.5 (`qwen2.5:7b`)

The evaluator scores novelty, feasibility, impact, and presentation, averages the three local model scores, flags disagreement for human review, ranks submissions, and writes feedback for finalists. By default, it produces one feedback response per finalist to keep local execution practical; three separate model feedback responses are available as an option.

## Ollama (primary)

1. Install Ollama from https://ollama.com.
2. Download the three models:

```powershell
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull qwen2.5:7b
```

3. Install Python dependencies and run the local smoke test:

```powershell
pip install -r requirements.txt
python test.py
```

4. Run the evaluator:

```powershell
python scorer.py
```

The default Ollama endpoint is `http://127.0.0.1:11434`. The input is `ideas.xlsx`; output is `ideas_scored.xlsx`.

## LM Studio (optional)

Load local Llama, Mistral, and Qwen models in LM Studio, start its local server, then create `.env` from `.env.example` and set:

```env
LLM_BACKEND=lm_studio
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LLAMA_MODEL=your-loaded-llama-model-id
MISTRAL_MODEL=your-loaded-mistral-model-id
QWEN_MODEL=your-loaded-qwen-model-id
```

LM Studio exposes an OpenAI-compatible local HTTP interface, but this project uses it only on `localhost`; it does not use an OpenAI account, API key, or hosted API.

## Configuration

Copy `.env.example` to `.env` only if you want to override defaults. Useful settings include `LLM_BACKEND`, local server URLs, model IDs, `LLM_TIMEOUT_SECONDS`, `INPUT_FILE`, and `OUTPUT_FILE`.

If you want the script to pick whichever local server is reachable, set:

```env
LLM_BACKEND=auto
```

`FEEDBACK_MODE=single` is the default and makes one feedback request for each finalist. Set `FEEDBACK_MODE=per_model` when you need separate feedback from Llama, Mistral, and Qwen; this triples feedback generation work.

Scores are checkpointed to `ideas_scored.xlsx` after every idea by default, so a stopped run retains completed scores. Set `CHECKPOINT_EVERY` to a larger number if Excel writes become a bottleneck.

GPU memory is the limiting factor when serving multiple models. For constrained hardware, use smaller Ollama model tags or run models sequentially by reducing the executor worker count in `scorer.py`.
