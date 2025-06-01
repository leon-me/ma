import pandas as pd
import numpy as np
from typing import Tuple, Set
from collections import Counter, defaultdict
import warnings


def calc_hitrate(row, documents, cluster_col):
    if not isinstance(row["original_doc_ids"], list):
        return None
    if not isinstance(row[cluster_col], list):
        raise RuntimeError(f"Values in cluster_col must be of type list. Received {type(row[cluster_col])}")

    hits = []

    for id in row["original_doc_ids"]:
        clusters = set(row[cluster_col])

        row_original_doc = documents.loc[documents["doc_id"].astype(int) == int(id)].iloc[0]
        clusters_original_doc = set(row_original_doc[cluster_col])

        hits.append(len(clusters.intersection(clusters_original_doc)) > 0)

    return np.mean(hits)


# values in cluster_col must be of type list
def evaluate_clusters(df: pd.DataFrame, cluster_col: str, return_recall_without_tabular: bool = False) -> Tuple:
    """
    Evaluates the DataFrame on the column specified and calculates Recall.

    Parameters:
    - df: pandas DataFrame containing the data.
    - cluster_col: str specifying the name of the column to evaluate on

    Returns:
    - out: Tuple(DataFrame, Recall, (Recall_without_tabular_docs)) with the modified DataFrame and the Recall over all Documents
    """
    df[f"{cluster_col}_hitrate"] = df.apply(calc_hitrate, args=(df, cluster_col), axis=1)

    hits_over_all_documents = []
    for _, row in df.iterrows():
        if not isinstance(row["original_doc_ids"], list):
            continue
        for _ in row["original_doc_ids"]:
            hits_over_all_documents.append(row[f"{cluster_col}_hitrate"])
    recall = np.mean(hits_over_all_documents)

    if return_recall_without_tabular:
        hits_over_non_tabular = []
        for _, row in df[~df["doc_id"].astype(str).str.startswith("3")].iterrows():
            if not isinstance(row["original_doc_ids"], list):
                continue
            for id in row["original_doc_ids"]:
                hits_over_non_tabular.append(row[f"{cluster_col}_hitrate"])
        recall_without_tabular = np.mean(hits_over_non_tabular)
        out = (df, recall, recall_without_tabular)
        return out

    out = (df, recall)

    return out


def count_documents_per_cluster(df: pd.DataFrame, cluster_col: str) -> float:
    counter = Counter()
    for membership in df[cluster_col]:
        counter.update(membership)

    return np.mean(list(counter.values()))


def count_avg_related_docs(df: pd.DataFrame, cluster_col: str) -> float:
    """
    Calculates the average number of comparisons needed based on shared cluster memberships.

    Parameters:
    - df: pandas DataFrame containing the data.
    - cluster_col: name of the column that contains a list/set of cluster memberships for each row.

    Returns:
    - Average number of comparisons per row.
    """
    # Build a mapping from cluster label to the set of document indices in that cluster
    cluster_to_docs: dict[str, Set[int]] = defaultdict(set)
    for idx, clusters in df[cluster_col].items():
        for label in clusters:
            cluster_to_docs[label].add(idx)

    # For each document, get all documents that share at least one cluster
    num_comparisons = []
    for clusters in df[cluster_col]:
        related_docs = set()
        for label in clusters:
            related_docs.update(cluster_to_docs[label])
        num_comparisons.append(len(related_docs))

    return float(np.mean(num_comparisons))


def select_up_to_distance_threshold(distances, df, threshold):
    distances = np.array(distances)
    mask = (distances > 0) & (distances <= threshold)
    matching_indices = np.where(mask)[0]
    matching_doc_ids = [df.iloc[index]["doc_id"] for index in matching_indices]

    return matching_doc_ids


def evaluate_distances(
    df: pd.DataFrame, distances_col: str, threshold: float = 0.25, return_recall_without_tabular: bool = False
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluates the recall for embedding distances.

    Parameters:
    - df: pandas DataFrame containing the data.
    - distances_col: name of the distances column

    Returns:
    - result (Dataframe, recall, avg_comparisons) with the modified DataFrame and the Recall over all Documents
    """

    df["related_docs"] = df[distances_col].apply(select_up_to_distance_threshold, args=(df, threshold))
    with warnings.catch_warnings(action="ignore"):
        df["hitrate"] = df.apply(
            lambda row: np.nanmean(
                [id in row["related_docs"] for id in row["original_doc_ids"]]
                if isinstance(row["original_doc_ids"], list)
                else [np.nan]
            ),
            axis=1,
        )

    hits_over_all_documents = []
    for _, row in df.iterrows():
        if not isinstance(row["original_doc_ids"], list):
            continue
        for id in row["original_doc_ids"]:
            hits_over_all_documents.append(id in row["related_docs"])
    recall = np.mean(hits_over_all_documents)

    if return_recall_without_tabular:
        hits_over_non_tabular = []
        for _, row in df[~df["doc_id"].astype(str).str.startswith("3")].iterrows():
            if not isinstance(row["original_doc_ids"], list):
                continue
            for id in row["original_doc_ids"]:
                hits_over_non_tabular.append(id in row["related_docs"])
        recall_without_tabular = np.mean(hits_over_non_tabular)

    nums_related_docs = []
    for _, row in df.iterrows():
        nums_related_docs.append(len(row["related_docs"]))
    avg_comparisons = np.mean(nums_related_docs)

    result = (df, recall, avg_comparisons, recall_without_tabular)

    return result
