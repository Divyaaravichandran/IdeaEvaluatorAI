"""Smoke test each configured local model without touching the spreadsheet."""

from scorer import LLM_CLIENT, MODEL_CONFIGS


def main():
    LLM_CLIENT.health_check()
    print(f"Connected to local {LLM_CLIENT.backend} server.")
    for model_key, config in MODEL_CONFIGS.items():
        response = LLM_CLIENT.completion(config["model"], "Reply with exactly: local model ready", 20)
        print(f"{model_key} ({config['model']}): {response.strip()}")


if __name__ == "__main__":
    main()
