from typing import Dict, List, Literal
import pandas as pd


def get_queries_by_doc_id(
    doc_id: int, queries: pd.DataFrame, query_types: List[str] = ["Factual Question"]
) -> List[Dict]:
    related_queries: pd.DataFrame = queries[
        (queries["ground_truth.doc_ids"].apply(lambda doc_ids: int(doc_id) in doc_ids))
        & (queries["query.query_type"].isin(query_types))
    ]
    return related_queries


def make_qa_pairs(queries: pd.DataFrame) -> List[Dict]:
    qa_pairs = []
    for _, query in queries.iterrows():
        question = query["query.content"]
        answer = query["ground_truth.content"]
        qa_pairs.append({"question": question, "answer": answer})

    return qa_pairs


def get_id_for_manipulated_doc_or_query(original_doc_id: int, prefix_number: Literal[1, 2, 3, 4]) -> int:
    id_str = str(prefix_number) + str(original_doc_id).zfill(5)
    return int(id_str)
