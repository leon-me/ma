import os
from typing import Iterator

from docling.datamodel import document
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from utils.tokenizer import OpenAITokenizerWrapper

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

import duckdb

import pandas
from tenacity import (
    retry,
    wait_random_exponential,
    retry_if_exception_type,
    stop_after_attempt,
)

env_path = os.path.join(os.sep, "Users", "leon", ".env")
load_dotenv(env_path)


class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """

    filename: str | None
    title: str | None


# Get the OpenAI embedding function from lancedb
embedding_func = get_registry().get("openai").create(name="text-embedding-3-small")


class Chunks(LanceModel):
    text: str = embedding_func.SourceField()
    vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()  # type: ignore | Pass the embeddings-function to the database
    metadata: ChunkMetadata


def extract_document(file_path: str | os.PathLike) -> Document:
    """Returns langchain Document"""
    file_extension = os.path.splitext(file_path)[1]

    if file_extension in [".txt", ".md"]:
        with open(file_path, "r") as f:
            content = f.read()
        doc = Document(
            page_content=content,
            metadata={"filename": os.path.basename(file_path), "title": None},
        )
        return doc
    else:
        loader = DoclingLoader(
            file_path=file_path,
            export_type=ExportType.MARKDOWN,
        )
        doc = loader.load()[0]
        return doc


def get_chunks(
    document: Document, chunker: str, *, max_length=1024, chunk_overlap=128
) -> list:
    if chunker == "docling-openai":
        client = OpenAI()
        tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
        chunker = HybridChunker(
            tokenizer=tokenizer, max_tokens=max_length, merge_peers=False
        )
        chunk_iter = chunker.chunk(dl_doc=document.document)
        return list(chunk_iter)

    if chunker == "langchain-recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents([document])
        return chunks


def get_chunks_multi(
    documents: list[Document],
    chunker: str = "langchain-recursive",
    *,
    max_length=512,
    chunk_overlap=128,
) -> list:
    if chunker == "langchain-recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        return chunks


def vectorise_and_load(chunks: list[Document]) -> pandas.DataFrame:
    def batch_list(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    def add_to_table(table, chunks):
        table.add(chunks)

    # LANCEDB_DIR = "../data/lancedb"
    LANCEDB_DIR = "/Users/leon/Documents/study/MA/lancedb"

    TABLE_NAME = "climatepolicyradar_laws_eur_en"

    # Create a LanceDB database and table
    db = lancedb.connect(LANCEDB_DIR)
    table = db.create_table(TABLE_NAME, schema=Chunks, mode="create")

    for i, chunks_batch in enumerate(batch_list(chunks, 100)):
        print(f"--- Processing chunks batch {i} ---")
        processed_chunks = [
            {
                "text": chunk.page_content,
                "metadata": {
                    "filename": chunk.metadata["filename"],
                    "title": chunk.metadata["title"],
                },
            }
            for chunk in chunks_batch
        ]
        add_to_table(table, processed_chunks)

    return table.to_pandas()


# Initial approach. Load data from txt files for 'EUR' and 'English'
def extract_load_eur():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    path_to_data_dir = "../data/ClimatePolicyRadar/txt"

    files = os.listdir(path_to_data_dir)
    files.sort()
    # files = files[:3]

    print(f"No. of files to process: {len(files)}")
    print(f"First 10 file names: {files[:10]}")

    # if len(files) > 10:
    #     raise RuntimeError("too many")

    documents = []
    for file in files:
        file_path = os.path.join(path_to_data_dir, file)
        documents.append(extract_document(file_path))

    chunks = get_chunks_multi(documents)

    user_input = ""
    while user_input not in ["y", "n"]:
        user_input = input(
            f"Got {len(chunks)} chunks! Do you want to proceed and vectorise all chunks? (y/n) "
        )
    if user_input == "n":
        print("Process aborted.")
        quit()

    df_table = vectorise_and_load(chunks)
    df_table.to_csv("table.csv")
    print("Wrote table to 'table.csv'.")


if __name__ == "__main__":
    table_name_ind_eng = "ind_eng"

    db = duckdb.connect("/Users/leon/Documents/study/MA/duckdb/climatePolicyRadar.db")
    print("Connected to duckdb.")
    df = db.sql(
        f"""
    SELECT count(distinct "document_id")
    FROM {table_name_ind_eng} 
    """
    ).to_df()

    print(df)
