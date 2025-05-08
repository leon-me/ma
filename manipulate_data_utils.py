### helper functions/classes for manipulation
from typing import Tuple, List
from pydantic import BaseModel
import pandas as pd
import ast
from dotenv import load_dotenv

load_dotenv("/Users/leon/.env")


DOCUMENTS = pd.read_csv("data/DRAGONball/en/docs.csv")
QUERIES = pd.read_csv(
    "data/DRAGONball/en/queries_flattened.csv",
    converters={
        "ground_truth.doc_ids": ast.literal_eval,
        "ground_truth.keypoints": ast.literal_eval,
        "ground_truth.references": ast.literal_eval,
    },
)


class FactualQuestionResponse(BaseModel):
    text_new: str
    answer_new: str
    references_new: list[str]


def format_prompt_man_factual(user_prompt: str, text: str, question: str, answer: str, references: str) -> str:
    """Inserts dynamic information into the user prompt."""
    return user_prompt.format(text=text, question=question, answer=answer, references=references)


def format_prompt_keypoints(user_prompt: str, question: str, answer: str) -> str:
    return user_prompt.format(question=question, ground_truth=answer)


def get_query_ids_for_doc(doc_id: int, query_types: list[str] = ["Factual Question"]) -> list[int]:
    """Selects query_ids for queries related to that doc and with a specified type."""
    return QUERIES[
        QUERIES["ground_truth.doc_ids"].apply(lambda doc_ids: doc_id in doc_ids)
        & QUERIES["query.query_type"].isin(query_types)
    ]["query.query_id"].to_list()


def get_query_properties(
    query_id,
    properties: list = ["ground_truth.content", "ground_truth.keypoints", "ground_truth.references", "query.content"],
) -> Tuple:
    """Select columns for query_id from queries dataframe."""
    row = QUERIES[QUERIES["query.query_id"] == query_id]
    return tuple(row[prop].iloc[0] for prop in properties)
    # row[prop].iloc[0] returns a scalar value (at position 0) instead of a Series


def get_doc_text(doc_id: int) -> str:
    return DOCUMENTS[DOCUMENTS["doc_id"] == doc_id]["content"].iloc[0]


def select_queries_for_docs_to_manipulate() -> List:
    """Select related queries to docs"""
    import csv

    with open("docs_to_manipulate.csv", "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        reader.__next__()

        doc_ids_for_man = set()
        for row in reader:
            doc_ids_for_man.add(int(row[0]))

    def filter_for_doc_ids(row):
        intersect_ids = doc_ids_for_man.intersection(set(row["ground_truth.doc_ids"]))
        if len(intersect_ids) > 0:
            return row["query.query_id"]
        return None

    filtered_ids = QUERIES.apply(filter_for_doc_ids, axis=1).dropna().astype(int)

    return filtered_ids.to_list()


def get_query_by_id(query_id: int) -> pd.Series:
    return QUERIES[QUERIES["query.query_id"].astype(int) == query_id].iloc[0]


def get_doc_by_id(doc_id: int) -> pd.Series:
    return DOCUMENTS[DOCUMENTS["doc_id"].astype(int) == doc_id].iloc[0]


def openai_interface(system_prompt, user_prompt, response_format_pydantic):
    """execute openai LLM call"""
    from openai import OpenAI

    client = OpenAI()

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=response_format_pydantic,
        temperature=0,
    ).choices[0]

    return completion.message.parsed


def execute_man_fact_single(doc_id: int, query_id: int) -> Tuple:
    ### prepare LLM manipulation call
    from utils.utils import read_prompt

    PROMPT_TYPE = "manipulation_factual"
    prompt = [
        prompt for prompt in read_prompt("prompts/json/manipulate_docs.json") if prompt["prompt_type"] == PROMPT_TYPE
    ][0]

    text = get_doc_text(doc_id)
    answer, keypoints, references, question = get_query_properties(query_id)

    system_prompt = prompt["system_prompt"]
    user_prompt = format_prompt_man_factual(
        user_prompt=prompt["user_prompt"], text=text, answer=answer, question=question, references=references
    )
    return (system_prompt, user_prompt)
