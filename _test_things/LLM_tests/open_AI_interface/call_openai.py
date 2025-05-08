import openai
import time
import os
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.sep, "Users", "leon", ".env")


def call_openai(prompt_user: str, model: str = "gpt-4o-mini"):
    load_dotenv(ENV_PATH)
    if os.environ.get("OPENAI_API_KEY") is None:
        raise RuntimeError("OPENAI_API_KEY not found!")

    if model != "gpt-4o-mini":
        error_message = f"Model must be 'gpt-4o-mini! Model is '{model}'."
        raise RuntimeError(error_message)

    client = openai.OpenAI()

    start_time = time.monotonic()

    completion = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt_user}]
    )

    end_time = time.monotonic()
    execution_time = end_time - start_time

    return {"completion": completion, "execution_time": execution_time}
