from typing import Tuple, List, Literal
import json
import pandas as pd
import ast
import os


def get_queries() -> pd.DataFrame:
    return pd.read_csv(
        "DRAGONball/en/queries_flattened.csv",
        usecols=[
            "domain",
            "ground_truth.content",
            "ground_truth.doc_ids",
            "ground_truth.keypoints",
            "ground_truth.references",
            "query.content",
            "query.query_id",
            "query.query_type",
        ],
        dtype={"query.query_id": "Int64"},
        converters={
            "ground_truth.doc_ids": ast.literal_eval,
            "ground_truth.keypoints": ast.literal_eval,
            "ground_truth.references": ast.literal_eval,
        },
    )


def get_documents() -> pd.DataFrame:
    return pd.read_csv(
        "DRAGONball/en/docs.csv",
        usecols=["doc_id", "domain", "content", "company_name", "court_name", "hospital_patient_name"],
        dtype={"doc_id": "Int64", "hospital_patient_name": str, "court_name": str, "company_name": str},
    )


def get_documents_to_manipulate(
    manipulation_type: Literal[
        "single_textual_manipulation", "single_tabular_manipulation", "multi_textual_manipulation"
    ],
) -> List[int]:
    mapping = pd.read_csv(
        "docs_to_manipulate.csv",
        usecols=["doc_id", manipulation_type],
        dtype={"doc_id": "Int64", manipulation_type: "boolean"},
    )
    doc_ids = mapping[mapping[manipulation_type] == True]["doc_id"].to_list()
    return doc_ids


def get_prompt(name: str) -> Tuple:
    """Reads from JSON-file

    Parameters:
    - name: name of the prompt in the format "folder_within_json_folder/prompt_name", e.g. "comparison/check_for_contradictions"

    Returns:
    - prompts: Tuple(system_prompt, user_prompt)
    """
    path = "../prompts/json/" + name + ".json"

    with open(path, "r") as f:
        prompt_json = json.load(f)

    out = (prompt_json["system_prompt"], prompt_json["user_prompt"])

    return out


def file_has_been_manipulated(kind: Literal["doc", "query"], file_name: str, id: int):
    if kind == "doc":
        folder = "docs"
        id_col = "doc_id"
    if kind == "query":
        folder = "queries"
        id_col = "query.query_id"

    path = f"additional_data/{folder}/{file_name}.csv"

    try:
        is_present = id in pd.read_csv(path, usecols=[id_col], dtype={id_col: "Int64"})[id_col].to_list()
    except FileNotFoundError:
        is_present = False
    return is_present


def delete_existing_doc_and_queries(doc_id: int, query_ids: List[int], file_name: str):
    path_docs = f"additional_data/docs/{file_name}.csv"
    path_queries = f"additional_data/queries/{file_name}.csv"

    docs = pd.read_csv(path_docs, dtype={"doc_id": "Int64"})
    queries = pd.read_csv(path_queries, dtype={"query.query_id": "Int64"})

    docs_filtered = docs[~docs.apply(lambda row: row["doc_id"] == doc_id, axis=1)]
    print(f"Deleting {len(docs) - len(docs_filtered)} documents. ID = '{doc_id}'")

    queries_filtered = queries[~queries.apply(lambda row: row["query.query_id"] in query_ids, axis=1)]
    print(f"Deleting {len(queries) - len(queries_filtered)} queries. IDs = [{query_ids}]")

    docs_filtered.to_csv(path_docs, mode="w", index=False)
    queries_filtered.to_csv(path_queries, mode="w", index=False)


def easy_save_manipulated_doc(file_name: str, data: pd.Series):
    """
    Parameters:
    - file_name: e.g. "multi_textual_manipulations"
    """
    save_manipulated_doc(
        file_name=file_name,
        doc_id=data["doc_id"],
        original_doc_ids=data["original_doc_ids"],
        domain=data["domain"],
        content=data["content"],
        company_name=data.get("company_name", ""),
        court_name=data.get("court_name", ""),
        hospital_patient_name=data.get("hospital_patient_name", ""),
    )


def save_manipulated_doc(
    file_name: str,
    doc_id: int,
    original_doc_ids: List[int],
    domain: str,
    content: str,
    company_name: str | None = "",
    court_name: str | None = "",
    hospital_patient_name: str | None = "",
):
    """
    Parameters:
    - file_name: e.g. "multi_textual_manipulations"
    """
    if company_name is None:
        company_name = ""
    if court_name is None:
        court_name = ""
    if hospital_patient_name is None:
        hospital_patient_name = ""

    path = f"additional_data/docs/{file_name}.csv"
    try:
        docs = pd.read_csv(
            path,
            usecols=[
                "doc_id",
                "original_doc_ids",
                "domain",
                "content",
                "company_name",
                "court_name",
                "hospital_patient_name",
            ],
            dtype={"doc_id": "Int64", "hospital_patient_name": str, "court_name": str, "company_name": str},
            converters={"original_doc_ids": ast.literal_eval},
        )
    except FileNotFoundError:
        docs = pd.DataFrame(
            columns=[
                "doc_id",
                "original_doc_ids",
                "domain",
                "content",
                "company_name",
                "court_name",
                "hospital_patient_name",
            ]
        )

    new_doc = {
        "doc_id": doc_id,
        "original_doc_ids": original_doc_ids,
        "domain": domain,
        "content": content,
        "company_name": company_name,
        "court_name": court_name,
        "hospital_patient_name": hospital_patient_name,
    }

    docs: pd.DataFrame = pd.concat([docs, pd.DataFrame([new_doc])], ignore_index=True)
    docs.to_csv(path, mode="w", index=False)
    print(f"Saved document to '{path}'")


def easy_save_manipulated_query(file_name: str, data: pd.Series):
    """
    Parameters:
    - file_name: e.g. "multi_textual_manipulations"
    """
    save_manipulated_query(
        file_name=file_name,
        domain=data["domain"],
        ground_truth_content=data["ground_truth.content"],
        ground_truth_doc_ids=data["ground_truth.doc_ids"],
        ground_truth_keypoints=data["ground_truth.keypoints"],
        ground_truth_references=data["ground_truth.references"],
        query_content=data["query.content"],
        query_query_id=data["query.query_id"],
        query_original_query_id=data["query.original_query_id"],
        query_query_type=data["query.query_type"],
    )


def save_manipulated_query(
    file_name: str,
    domain: str,
    ground_truth_content: str,
    ground_truth_doc_ids: List[int],
    ground_truth_keypoints: List[str],
    ground_truth_references: List[str],
    query_content: str,
    query_query_id: int,
    query_original_query_id: int,
    query_query_type: str,
):
    """
    Parameters:
    - file_name: e.g. "multi_textual_manipulations"
    """
    path = f"additional_data/queries/{file_name}.csv"
    try:
        queries = pd.read_csv(
            path,
            usecols=[
                "domain",
                "ground_truth.content",
                "ground_truth.doc_ids",
                "ground_truth.keypoints",
                "ground_truth.references",
                "query.content",
                "query.query_id",
                "query.original_query_id",
                "query.query_type",
            ],
            dtype={"query.query_id": "Int64", "query.original_query_id": "Int64"},
            converters={
                "ground_truth.doc_ids": ast.literal_eval,
                "ground_truth.keypoints": ast.literal_eval,
                "ground_truth.references": ast.literal_eval,
            },
        )
    except FileNotFoundError:
        queries = pd.DataFrame(
            columns=[
                "domain",
                "ground_truth.content",
                "ground_truth.doc_ids",
                "ground_truth.keypoints",
                "ground_truth.references",
                "query.content",
                "query.query_id",
                "query.original_query_id",
                "query.query_type",
            ]
        )
    new_query = {
        "domain": domain,
        "ground_truth.content": ground_truth_content,
        "ground_truth.doc_ids": ground_truth_doc_ids,
        "ground_truth.keypoints": ground_truth_keypoints,
        "ground_truth.references": ground_truth_references,
        "query.content": query_content,
        "query.query_id": query_query_id,
        "query.original_query_id": query_original_query_id,
        "query.query_type": query_query_type,
    }
    queries = pd.concat([queries, pd.DataFrame([new_query])], ignore_index=True)
    queries.to_csv(path, mode="w", index=False)

    print(f"Saved Query to '{path}'")
