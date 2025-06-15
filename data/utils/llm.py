from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict


class KeypointsGenerationResponse(BaseModel):
    keypoints: list[str]


class SingleTextualManipulationResponse(BaseModel):
    text_new: str
    answer_new: str
    references_new: list[str]


class QAPair(BaseModel):
    question: str
    answer: str
    references: list[str]


class QAPairV02(BaseModel):
    question: str
    answer: str
    quote: str


class MultiTextualManipulationResponse(BaseModel):
    text_new: str
    qa_pairs_new: list[QAPair]


class MultiTextualManipulationResponseV02(BaseModel):
    text_new: str
    qa_pairs_new: list[QAPairV02]


class SingleTabularManipulationResponse(BaseModel):
    answer_new: str
    description: str
    value: str


def format_user_prompt_single_textual(user_prompt: str, text: str, question: str, answer: str, references: str) -> str:
    return user_prompt.format(text=text, question=question, answer=answer, references=references)


def format_user_prompt_single_tabular(user_prompt: str, question: str, answer: str, entity: str) -> str:
    return user_prompt.format(question=question, answer=answer, entity=entity)


def format_user_prompt_multi_textual(user_prompt: str, text: str, qa_pairs: list[Dict]) -> str:
    """qa_pairs must be of format: [ { question: '...', answer: '...' }, ... ]"""
    questions_str = ""
    for id, qa in enumerate(qa_pairs, start=1):
        questions_str += f"question_{id}: {qa["question"]}\n"
        questions_str += f"answer_{id}: {qa["answer"]}\n"

    return user_prompt.format(text=text, questions=questions_str)


def format_user_prompt_multi_textual_v02(user_prompt: str, entity: str, text: str, qa_pairs: list[Dict]) -> str:
    """format new prompt version of multi_textual

    Parameters:
    - user_prompt: template to insert values into
    - entity: name of patient, company or court
    - text: content of document
    - qa_pairs: must be of format: [ { question: '...', answer: '...' }, ... ]

    Returns:
    - prompt: formatted user prompt
    """

    questions_str = ""
    for id, qa in enumerate(qa_pairs, start=1):
        questions_str += f"*question_{id}*: {qa["question"]}\n"
        questions_str += f"*answer_{id}*: {qa["answer"]}\n"

    return user_prompt.format(text=text, entity=entity, questions=questions_str)


def format_user_prompt_keypoints(user_prompt: str, question: str, answer: str):
    return user_prompt.format(question=question, ground_truth=answer)


def call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    response_format_pydantic: type[BaseModel],
    temperature: float = 0.0,
) -> BaseModel:
    if model is None:
        raise RuntimeError("Model parameter not set!")
    if response_format_pydantic is None:
        raise RuntimeError("response_format_pydantic parameter not set!")

    client = OpenAI()

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=response_format_pydantic,
        temperature=temperature,
    )

    return completion.choices[0].message.parsed
