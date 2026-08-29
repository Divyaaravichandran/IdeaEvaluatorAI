"""Evaluate hackathon ideas with a local, open-source LLM ensemble."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = os.getenv("INPUT_FILE", "ideas.xlsx")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "ideas_scored.xlsx")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
SCORING_MAX_TOKENS = int(os.getenv("SCORING_MAX_TOKENS", "180"))
HUMAN_REVIEW_SPREAD_THRESHOLD = 0.75
ENSEMBLE_METHOD = os.getenv("ENSEMBLE_METHOD", "median").lower()
MODEL_WEIGHTS = {
    "novelty": 0.20,
    "feasibility": 0.30,
    "impact": 0.35,
    "presentation": 0.15,
}

# All three defaults are freely downloadable open-weight models from Ollama.
MODEL_CONFIGS = {
    "llama": {"model": os.getenv("LLAMA_MODEL", "llama3.2:3b")},
    "mistral": {"model": os.getenv("MISTRAL_MODEL", "mistral:7b")},
    "qwen": {"model": os.getenv("QWEN_MODEL", "qwen2.5:7b")},
}
LEGACY_MODEL_KEYS = ("qwen3_32b", "deepseek")
FEEDBACK_MODE = os.getenv("FEEDBACK_MODE", "single").lower()
FEEDBACK_MODEL_KEY = os.getenv("FEEDBACK_MODEL_KEY", "llama")
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "1"))


class LocalLLMClient:
    """Small adapter for Ollama and LM Studio's local HTTP servers."""

    def __init__(self):
        self.backend = os.getenv("LLM_BACKEND", "ollama").lower()
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.lm_studio_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        if self.backend not in {"ollama", "lm_studio", "auto"}:
            raise ValueError("LLM_BACKEND must be 'ollama', 'lm_studio', or 'auto'.")
        if self.backend == "auto":
            self.backend = self._detect_backend()

    def _detect_backend(self):
        last_error = None
        for backend, checker in (("ollama", self._check_ollama), ("lm_studio", self._check_lm_studio)):
            try:
                checker()
                return backend
            except requests.RequestException as error:
                last_error = error
        raise RuntimeError(
            "Could not find a reachable local LLM server. Start Ollama on "
            f"{self.ollama_url} or LM Studio on {self.lm_studio_url}. "
            f"Last connection error: {last_error}"
        ) from last_error

    def _check_ollama(self):
        response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
        response.raise_for_status()

    def _check_lm_studio(self):
        response = requests.get(f"{self.lm_studio_url}/models", timeout=10)
        response.raise_for_status()

    def completion(self, model, prompt, max_tokens=80):
        if self.backend == "ollama":
            return self._ollama_completion(model, prompt, max_tokens)
        return self._lm_studio_completion(model, prompt, max_tokens)

    def health_check(self):
        if self.backend == "ollama":
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
        else:
            response = requests.get(f"{self.lm_studio_url}/models", timeout=10)
        response.raise_for_status()

    def _ollama_completion(self, model, prompt, max_tokens):
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content")
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content

    def _lm_studio_completion(self, model, prompt, max_tokens):
        response = requests.post(
            f"{self.lm_studio_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE,
                "max_tokens": max_tokens,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        choices = response.json().get("choices", [])
        content = choices[0].get("message", {}).get("content") if choices else None
        if not content:
            raise RuntimeError("LM Studio returned an empty response.")
        return content


LLM_CLIENT = LocalLLMClient()


def build_prompt(title, description):
    return f"""You are a strict expert hackathon judge. Evaluate this idea on four criteria from 1 to 4.

Score 4 only for exceptional ideas, 3 for good ideas, 2 for average ideas, and 1 for weak or common ideas.
Novelty: originality. Feasibility: can a student team build it in 24-48 hours. Impact: real-world value. Presentation: clarity of the proposal.

Advancement guidance: an idea is strongest when its weighted overall score is at least 3/4,
with feasibility and impact also at least 3/4. Apply these standards consistently to every idea.

Project title: {title}
Project description: {description}

Reply with only four numbers separated by semicolons in this exact order:
novelty;feasibility;impact;presentation
Example: 2;3;2;2"""


def build_feedback_prompt(title, description, scores):
    novelty, feasibility, impact, presentation = scores
    return f"""You are a hackathon mentor.
Project title: {title}
Project description: {description}
Scores: novelty={novelty}/4, feasibility={feasibility}/4, impact={impact}/4, presentation={presentation}/4.

Give exactly three short, specific, actionable suggestions to improve the weakest areas.
Each suggestion must be one numbered item and no more than 35 words.
Number them 1, 2, and 3."""


def parse_scores(text):
    if not isinstance(text, str):
        return None
    if "</think>" in text:
        text = text.split("</think>")[-1]
    matches = re.findall(r"\b([1-4](?:\.\d+)?)\b", text.replace("\n", " "))
    if len(matches) < 4:
        return None
    scores = [float(value) for value in matches[:4]]
    return scores if all(1 <= score <= 4 for score in scores) else None


def calculate_overall(scores):
    return round(sum(score * MODEL_WEIGHTS[criterion] for score, criterion in zip(scores, MODEL_WEIGHTS)), 2)


def complete_model(model_key, prompt, max_tokens=80):
    return LLM_CLIENT.completion(MODEL_CONFIGS[model_key]["model"], prompt, max_tokens)


def score_model(model_key, title, description):
    try:
        scores = parse_scores(
            complete_model(
                model_key,
                build_prompt(title, description),
                max_tokens=SCORING_MAX_TOKENS,
            )
        )
        if scores is None:
            raise ValueError("model did not return four valid scores in the 1-4 range")
        return model_key, scores, "completed"
    except Exception as error:
        return model_key, None, f"failed: {error}"


def write_model_scores(dataframe, row_index, model_key, scores, status):
    dataframe.at[row_index, f"{model_key}_status"] = status
    if scores is not None:
        for criterion, score in zip(MODEL_WEIGHTS, scores):
            dataframe.at[row_index, f"{model_key}_{criterion}"] = score
        dataframe.at[row_index, f"{model_key}_overall"] = calculate_overall(scores)


def assign_stable_ranks(dataframe, score_column, rank_column):
    ordered_indices = dataframe.sort_values([score_column, "idea_id"], ascending=[False, True], na_position="last").index
    dataframe[rank_column] = pd.NA
    dataframe.loc[ordered_indices, rank_column] = range(1, len(dataframe) + 1)
    dataframe[rank_column] = dataframe[rank_column].astype("Int64")


def select_top_k(dataframe, score_column, number_to_select, allow_shortfall=False):
    valid_scores = dataframe.dropna(subset=[score_column]).sort_values([score_column, "idea_id"], ascending=[False, True])
    if len(valid_scores) >= number_to_select:
        return valid_scores.head(number_to_select).index
    if allow_shortfall:
        print(f"Warning: only {len(valid_scores)} valid {score_column} scores are available.")
        return valid_scores.index
    raise RuntimeError(f"Only {len(valid_scores)} valid {score_column} scores are available; cannot select {number_to_select} finalists.")


def calculate_metrics(dataframe, predictions, label_column="advance"):
    actual, predicted = dataframe[label_column].astype(int), predictions.astype(int)
    tp = int(((predicted == 1) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    accuracy = (tp + tn) / len(dataframe) if len(dataframe) else 0
    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "balanced_accuracy": (recall + specificity) * 50,
    }


def save_results(dataframe):
    try:
        dataframe.to_excel(OUTPUT_FILE, index=False)
    except PermissionError as error:
        raise RuntimeError(f"Cannot save {OUTPUT_FILE}. Close it in Excel and run the script again.") from error


def score_all_ideas():
    try:
        LLM_CLIENT.health_check()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Cannot connect to local {LLM_CLIENT.backend} server. "
            f"Check that Ollama is running on {LLM_CLIENT.ollama_url} "
            f"or set LLM_BACKEND=lm_studio and point LM_STUDIO_BASE_URL to a running server. "
            f"Original error: {error}"
        ) from error

    dataframe = pd.read_excel(INPUT_FILE)
    required_columns = {"idea_id", "title", "description", "advance", "expert_rank"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{INPUT_FILE} is missing required columns: {sorted(missing_columns)}")
    if FEEDBACK_MODE not in {"single", "per_model"}:
        raise ValueError("FEEDBACK_MODE must be either 'single' or 'per_model'.")
    if FEEDBACK_MODEL_KEY not in MODEL_CONFIGS:
        raise ValueError(f"FEEDBACK_MODEL_KEY must be one of: {', '.join(MODEL_CONFIGS)}.")
    if ENSEMBLE_METHOD not in {"mean", "median"}:
        raise ValueError("ENSEMBLE_METHOD must be either 'mean' or 'median'.")
    if CHECKPOINT_EVERY < 1:
        raise ValueError("CHECKPOINT_EVERY must be at least 1.")

    # Keep migrated exports free of columns from the former hosted-model setup.
    legacy_columns = [
        column
        for column in dataframe.columns
        if column.startswith(tuple(f"{key}_" for key in LEGACY_MODEL_KEYS))
    ]
    dataframe = dataframe.drop(columns=legacy_columns)

    for model_key in MODEL_CONFIGS:
        dataframe[f"{model_key}_status"] = pd.NA
        dataframe[f"{model_key}_feedback"] = pd.NA
        for criterion in (*MODEL_WEIGHTS, "overall"):
            dataframe[f"{model_key}_{criterion}"] = pd.NA
    dataframe["combined_feedback"] = pd.NA

    print(f"Evaluating {len(dataframe)} ideas using local {LLM_CLIENT.backend}: {', '.join(config['model'] for config in MODEL_CONFIGS.values())}")
    for position, (row_index, row) in enumerate(dataframe.iterrows(), start=1):
        print(f"Idea {position}/{len(dataframe)}: {row['title']}")
        with ThreadPoolExecutor(max_workers=len(MODEL_CONFIGS)) as executor:
            futures = [executor.submit(score_model, key, row["title"], row["description"]) for key in MODEL_CONFIGS]
            for future in as_completed(futures):
                model_key, scores, status = future.result()
                write_model_scores(dataframe, row_index, model_key, scores, status)
                print(f"  {model_key}: {calculate_overall(scores) if scores else status}")
        if position % CHECKPOINT_EVERY == 0 or position == len(dataframe):
            save_results(dataframe)
            print(f"  Checkpoint saved after idea {position}.")

    score_columns = [f"{key}_overall" for key in MODEL_CONFIGS]
    dataframe[score_columns] = dataframe[score_columns].apply(pd.to_numeric, errors="coerce")
    if ENSEMBLE_METHOD == "median":
        dataframe["avg_overall"] = dataframe[score_columns].median(axis=1).round(2)
    else:
        dataframe["avg_overall"] = dataframe[score_columns].mean(axis=1).round(2)
    dataframe["score_spread"] = (dataframe[score_columns].max(axis=1) - dataframe[score_columns].min(axis=1)).round(2)
    dataframe["human_review_required"] = ((dataframe[score_columns].notna().sum(axis=1) < len(MODEL_CONFIGS)) | (dataframe["score_spread"] >= HUMAN_REVIEW_SPREAD_THRESHOLD)).astype(int)
    for model_key in MODEL_CONFIGS:
        assign_stable_ranks(dataframe, f"{model_key}_overall", f"{model_key}_rank")
    assign_stable_ranks(dataframe, "avg_overall", "final_rank")

    finalists_to_select = int(dataframe["advance"].sum())
    final_indices = select_top_k(dataframe, "avg_overall", finalists_to_select)
    dataframe["ai_advance"] = 0
    dataframe.loc[final_indices, "ai_advance"] = 1
    dataframe["agrees_with_expert"] = (dataframe["ai_advance"] == dataframe["advance"].astype(int)).astype(int)

    feedback_keys = list(MODEL_CONFIGS) if FEEDBACK_MODE == "per_model" else [FEEDBACK_MODEL_KEY]
    print(f"Generating {FEEDBACK_MODE} feedback for {finalists_to_select} finalists using: {', '.join(feedback_keys)}")
    for row_index in final_indices:
        row = dataframe.loc[row_index]
        with ThreadPoolExecutor(max_workers=len(MODEL_CONFIGS)) as executor:
            futures = {
                executor.submit(
                    complete_model,
                    key,
                    build_feedback_prompt(
                        row["title"],
                        row["description"],
                        [row[f"{key}_{criterion}"] for criterion in MODEL_WEIGHTS],
                    ),
                    160,
                ): key
                for key in feedback_keys
                if dataframe.at[row_index, f"{key}_status"] == "completed"
            }
            for future in as_completed(futures):
                model_key = futures[future]
                try:
                    feedback = future.result().strip()
                    dataframe.at[row_index, f"{model_key}_feedback"] = feedback
                    if FEEDBACK_MODE == "single":
                        dataframe.at[row_index, "combined_feedback"] = feedback
                except Exception as error:
                    feedback = f"Feedback unavailable: {error}"
                    dataframe.at[row_index, f"{model_key}_feedback"] = feedback
                    if FEEDBACK_MODE == "single":
                        dataframe.at[row_index, "combined_feedback"] = feedback

    metrics_by_model = {}
    for model_key in MODEL_CONFIGS:
        predictions = pd.Series(0, index=dataframe.index)
        predictions.loc[select_top_k(dataframe, f"{model_key}_overall", finalists_to_select, allow_shortfall=True)] = 1
        metrics_by_model[model_key] = calculate_metrics(dataframe, predictions)
    metrics_by_model["combined"] = calculate_metrics(dataframe, dataframe["ai_advance"])
    for model_key, metrics in metrics_by_model.items():
        dataframe[f"{model_key}_accuracy"] = metrics["accuracy"]
    correlation = dataframe[["final_rank", "expert_rank"]].corr(method="spearman").iloc[0, 1]
    dataframe["spearman_rank_correlation"] = correlation
    save_results(dataframe)

    print(f"Results saved to {OUTPUT_FILE}. Human review required: {int(dataframe['human_review_required'].sum())}. Spearman correlation: {correlation:.3f}")
    for model_key, metrics in metrics_by_model.items():
        print(f"{model_key:<10} accuracy={metrics['accuracy']:.1f}% precision={metrics['precision']:.1f}% recall={metrics['recall']:.1f}% f1={metrics['f1']:.1f}% balanced={metrics['balanced_accuracy']:.1f}%")


if __name__ == "__main__":
    score_all_ideas()
