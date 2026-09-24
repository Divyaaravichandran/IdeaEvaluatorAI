# AIEval — Hackathon Idea Evaluation Dashboard

AIEval evaluates hackathon ideas with three locally hosted open-weight language models, ranks them against expert decisions, generates improvement feedback, and presents the results in a Streamlit dashboard.

The project runs locally and does not require an OpenAI account, API key, paid provider, or hosted AI service.

## Features

- Scores novelty, feasibility, impact, and presentation.
- Uses weighted scoring: novelty 20%, feasibility 30%, impact 35%, presentation 15%.
- Evaluates ideas with Llama, Mistral, and Qwen.
- Combines scores using a configurable mean or median ensemble.
- Selects the same number of AI finalists as expert-advanced ideas.
- Calculates accuracy, precision, recall, F1, balanced accuracy, and rank correlation.
- Flags model disagreement for human review.
- Generates three concise improvement suggestions for finalists.
- Provides a responsive dashboard with Home, Leaderboard, Model Comparison, Idea Detail, and Live Evaluation Studio pages.

## Default models

```text
llama3.2:3b
mistral:7b
qwen2.5:7b
```

## Requirements

- Python 3.10 or newer recommended
- Ollama or LM Studio
- Sufficient RAM/VRAM for the selected models
- An input Excel file with:

```text
idea_id, title, description, advance, expert_rank
```

Optional `category` and `source` columns are displayed when available.

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

Load local Llama, Mistral, and Qwen models in LM Studio, start its local server, then set these values in an optional `.env` file:

```env
LLM_BACKEND=lm_studio
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LLAMA_MODEL=your-loaded-llama-model-id
MISTRAL_MODEL=your-loaded-mistral-model-id
QWEN_MODEL=your-loaded-qwen-model-id
```

LM Studio exposes an OpenAI-compatible local HTTP interface, but this project uses it only on `localhost`; it does not use an OpenAI account, API key, or hosted API.

## Configuration

Create an optional `.env` file to override defaults. Useful settings include `LLM_BACKEND`, local server URLs, model IDs, `LLM_TIMEOUT_SECONDS`, `INPUT_FILE`, and `OUTPUT_FILE`.

If you want the script to pick whichever local server is reachable, set:

```env
LLM_BACKEND=auto
```

`FEEDBACK_MODE=single` is the default and makes one feedback request for each finalist. Set `FEEDBACK_MODE=per_model` when you need separate feedback from Llama, Mistral, and Qwen; this triples feedback generation work.

Scores are checkpointed to `ideas_scored.xlsx` after every idea by default, so a stopped run retains completed scores. Set `CHECKPOINT_EVERY` to a larger number if Excel writes become a bottleneck.

GPU memory is the limiting factor when serving multiple models. For constrained hardware, use smaller Ollama model tags or run models sequentially by reducing the executor worker count in `scorer.py`.
