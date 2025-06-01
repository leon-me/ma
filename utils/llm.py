from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv("/Users/leon/.env")


class LLMDocumentComparisonResponse(BaseModel):
    contradictory_info_found: bool


def format_user_prompt(prompt: str, text1: str, text2: str) -> str:
    return prompt.format(text1=text1, text2=text2)


def call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    response_format_pydantic: type[BaseModel] = LLMDocumentComparisonResponse,
    temperature: float = 0.0,
) -> BaseModel:
    if model is None:
        raise RuntimeError("Model parameter not set!")

    client = OpenAI()

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=response_format_pydantic,
        temperature=temperature,
    )

    return completion.choices[0].message.parsed
