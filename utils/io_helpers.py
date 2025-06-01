import os
import json
from typing import List, Dict
import pandas as pd
import ast


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
    docs_manipulated_single_textual = pd.read_csv(
        "data/additional_data/docs/textual_manipulations_result.csv",
        usecols=["doc_id", "domain", "content", "original_doc_id"],
        dtype={"original_doc_id": "Int64"},
    )
    docs_manipulated_single_textual["original_doc_id"] = docs_manipulated_single_textual["original_doc_id"].apply(
        lambda i: [i] if pd.notna(i) else []
    )
    docs_manipulated_single_textual.rename(columns={"original_doc_id": "original_doc_ids"}, inplace=True)

    docs_manipulated_single_tabular = pd.read_csv(
        "data/additional_data/docs/tabular_manipulations_result.csv",
        usecols=["doc_id", "domain", "content", "original_doc_ids"],
        converters={"original_doc_ids": ast.literal_eval},
    )

    docs_manipulated_multi_textual = pd.read_csv(
        "data/additional_data/docs/multi_textual_manipulations.csv",
        usecols=["doc_id", "domain", "content", "original_doc_id"],
        dtype={"original_doc_id": "Int64"},
    )
    docs_manipulated_multi_textual["original_doc_id"] = docs_manipulated_multi_textual["original_doc_id"].apply(
        lambda i: [i] if pd.notna(i) else []
    )
    docs_manipulated_multi_textual.rename(columns={"original_doc_id": "original_doc_ids"}, inplace=True)

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
