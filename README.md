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

## Setup with Ollama

Install Ollama from [ollama.com](https://ollama.com), then download the models:

```powershell
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull qwen2.5:7b
```

Install dependencies and run the smoke test:

```powershell
pip install -r requirements.txt
python test.py
```

## Run the evaluator

Place the source dataset at `ideas.xlsx`, or configure `INPUT_FILE`, then run:

```powershell
python scorer.py
```

Results are written to `ideas_scored.xlsx`. The script checkpoints the workbook after each idea by default.

## Run the dashboard

After scoring, start Streamlit from the project directory:

```powershell
streamlit run app.py
```

The dashboard reads `ideas_scored.xlsx`. Regenerate the workbook and restart or refresh Streamlit after a new scoring run.


## Project files

```text
app.py              Streamlit dashboard and live evaluation UI
scorer.py           Model scoring, ensemble ranking, metrics, and feedback
test.py             Local backend smoke test
create_template.py  Creates a sample input workbook
ideas.xlsx          Input idea dataset
ideas_scored.xlsx   Generated scoring results
requirements.txt    Python dependencies
```

## Troubleshooting

- If the dashboard shows old values, confirm that `ideas_scored.xlsx` was regenerated in the same project directory and restart Streamlit.
- If a model fails, verify that it is downloaded or loaded and reachable at the configured local endpoint.
- If generation is slow, use smaller model tags, increase `LLM_TIMEOUT_SECONDS`, or reduce concurrent model execution.
- If feedback is cut off, increase the feedback generation limit in the calling code.

