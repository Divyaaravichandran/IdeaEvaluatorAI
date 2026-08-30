# Hackathon Idea Evaluation and Ranking System

This project evaluates and ranks hackathon ideas with a local ensemble of open-weight LLMs. It makes no calls to paid or hosted AI providers and requires no API keys.

## Default model ensemble

- Llama 3.2 (`llama3.2:3b`)
- Mistral (`mistral:7b`)
- Qwen 2.5 (`qwen2.5:7b`)

The evaluator scores novelty, feasibility, impact, and presentation, averages the three local model scores, flags disagreement for human review, ranks submissions, and writes feedback for finalists. By default, it produces one feedback response per finalist to keep local execution practical; three separate model feedback responses are available as an option.


