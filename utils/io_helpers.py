import os
import json
from typing import Dict, Literal
import pandas as pd
import ast


def is_csv_empty(file_path) -> bool:
    return os.stat(file_path).st_size == 0


def get_prompt(name: str) -> Dict:
    """Reads from JSON-file

    Parameters:
    - name: name of the prompt in the format "folder_within_json_folder/prompt_name", e.g. "comparison/check_for_contradictions"
    """
    path = "prompts/json/" + name + ".json"

    with open(path, "r") as f:
        return json.load(f)


def get_documents(
    print_info: bool = False, read_embeddings: bool = False, read_relations: bool = False
) -> pd.DataFrame:
    docs_original = pd.read_csv("data/DRAGONball/en/docs.csv", usecols=["doc_id", "domain", "content"])
    docs_original["original_doc_ids"] = docs_original.apply(lambda _: [], axis=1)
    docs_manipulated_single_textual = pd.read_csv(
        "data/additional_data/docs/single_textual_manipulations.csv",
        usecols=["doc_id", "domain", "content", "original_doc_ids"],
        converters={"original_doc_ids": ast.literal_eval},
    )

    docs_manipulated_single_tabular = pd.read_csv(
        "data/additional_data/docs/single_tabular_manipulations.csv",
        usecols=["doc_id", "domain", "content", "original_doc_ids"],
        converters={"original_doc_ids": ast.literal_eval},
    )

    docs_manipulated_multi_textual = pd.read_csv(
        "data/additional_data/docs/multi_textual_manipulations.csv",
        usecols=["doc_id", "domain", "content", "original_doc_ids"],
        converters={"original_doc_ids": ast.literal_eval},
    )

    if print_info == True:
        print(f"# original docs: {len(docs_original)}")
        print(f"# manipulated textual docs: {len(docs_manipulated_single_textual)}")
        print(f"# manipulated tabular docs: {len(docs_manipulated_single_tabular)}")
        print(f"# manipulated textual multi docs: {len(docs_manipulated_multi_textual)}")
        print(
            f"= {len(docs_original) + len(docs_manipulated_single_textual) + len(docs_manipulated_single_tabular) + len(docs_manipulated_multi_textual)} documents in total"
        )

    result = pd.concat(
        [
            docs_original,
            docs_manipulated_single_textual,
            docs_manipulated_multi_textual,
            docs_manipulated_single_tabular,
        ],
        sort=False,
    )

    if read_embeddings:
        embeddings = pd.read_csv(
            "data/additional_data/docs/_embeddings.csv",
            usecols=["doc_id", "embedding"],
            converters={"embedding": ast.literal_eval},
        )
        result = pd.merge(result, embeddings, on="doc_id", how="left")

    if read_relations:
        relations = pd.read_csv(
            "data/additional_data/docs/_relations.csv",
            usecols=["doc_id", "related_docs"],
            converters={"related_docs": ast.literal_eval},
        )
        result = pd.merge(result, relations, on="doc_id", how="left")

    return result


def get_queries(filter: Literal["original_only", "manipulated_only"] | None = None) -> pd.DataFrame:
    queries_original = pd.read_csv(
        "data/DRAGONball/en/queries_flattened.csv",
        usecols=[
            "domain",
            "ground_truth.doc_ids",
            "ground_truth.content",
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
    queries_original["query.original_query_id"] = queries_original.apply(lambda _: [], axis=1)

    queries_manipulated_single_textual = pd.read_csv(
        "data/additional_data/queries/single_textual_manipulations.csv",
        usecols=[
            "domain",
            "ground_truth.doc_ids",
            "ground_truth.content",
            "ground_truth.keypoints",
            "ground_truth.references",
            "query.content",
            "query.query_id",
            "query.query_type",
            "query.original_query_id",
        ],
        dtype={"query.query_id": "Int64", "query.original_query_id": "Int64"},
        converters={
            "ground_truth.doc_ids": ast.literal_eval,
            "ground_truth.keypoints": ast.literal_eval,
            "ground_truth.references": ast.literal_eval,
        },
    )

    queries_manipulated_single_tabular = pd.read_csv(
        "data/additional_data/queries/single_tabular_manipulations.csv",
        usecols=[
            "domain",
            "ground_truth.doc_ids",
            "ground_truth.content",
            "ground_truth.keypoints",
            "ground_truth.references",
            "query.content",
            "query.query_id",
            "query.query_type",
            "query.original_query_id",
        ],
        dtype={"query.query_id": "Int64", "query.original_query_id": "Int64"},
        converters={
            "ground_truth.doc_ids": ast.literal_eval,
            "ground_truth.keypoints": ast.literal_eval,
            "ground_truth.references": ast.literal_eval,
        },
    )

    queries_manipulated_multi_textual = pd.read_csv(
        "data/additional_data/queries/multi_textual_manipulations.csv",
        usecols=[
            "domain",
            "ground_truth.doc_ids",
            "ground_truth.content",
            "ground_truth.keypoints",
            "ground_truth.references",
            "query.content",
            "query.query_id",
            "query.query_type",
            "query.original_query_id",
        ],
        dtype={"query.query_id": "Int64", "query.original_query_id": "Int64"},
        converters={
            "ground_truth.doc_ids": ast.literal_eval,
            "ground_truth.keypoints": ast.literal_eval,
            "ground_truth.references": ast.literal_eval,
        },
    )

    if filter == "original_only":
        result = queries_original
    elif filter == "manipulated_only":
        result = pd.concat(
            [
                queries_manipulated_single_textual,
                queries_manipulated_single_tabular,
                queries_manipulated_multi_textual,
            ],
            sort=False,
        )
    else:
        result = pd.concat(
            [
                queries_original,
                queries_manipulated_single_textual,
                queries_manipulated_multi_textual,
                queries_manipulated_single_tabular,
            ],
            sort=False,
        )
    return result
