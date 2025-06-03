from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv("/Users/leon/.env")


class LLMDocumentComparisonCheckResponse(BaseModel):
    contradictory_info_found: bool


class Contradiction(BaseModel):
    quote_from_document1: str
    quote_from_document2: str


class LLMDocumentComparisonExtractResponse(BaseModel):
    contradictory_info_found: bool
    contradictions: list[Contradiction]


def format_user_prompt(prompt: str, text1: str, text2: str) -> str:
    return prompt.format(text1=text1, text2=text2)


def call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    response_format_pydantic: type[BaseModel] = LLMDocumentComparisonCheckResponse,
    temperature: float = 0.0,
) -> BaseModel:
    if model is None:
        raise RuntimeError("Model parameter not set!")

    client = OpenAI()
    print("calling openai")
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=response_format_pydantic,
        temperature=temperature,
    )

    return completion.choices[0].message.parsed


def call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str,
    response_format_pydantic: type[BaseModel] = LLMDocumentComparisonCheckResponse,
    temperature: float = 0.0,
) -> BaseModel:
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=response_format_pydantic,
            temperature=temperature,
        ),
    )
    return response.parsed


def call_any_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    response_format_pydantic: type[BaseModel] = LLMDocumentComparisonCheckResponse,
    temperature: float = 0.0,
) -> type[BaseModel]:
    kwargs = locals()
    openai_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ]
    gemini_models = ["gemini-2.5-flash-preview-05-20"]
    if model in openai_models:
        return call_openai(**kwargs)
    elif model in gemini_models:
        return call_gemini(**kwargs)
    else:
        error_message = f"Model {model} not in accepted models!"
        raise RuntimeError(error_message)
