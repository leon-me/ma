from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper
import os

env_path = os.path.join(os.sep, "Users", "leon", ".env")
load_dotenv(env_path)

# Initialize OpenAI client (make sure you have OPENAI_API_KEY in your environment variables)
client = OpenAI()


tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length


# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

os.chdir(os.path.dirname(os.path.abspath(__file__)))

converter = DocumentConverter()
result = converter.convert("test.md")


# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------



chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)



chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

print(len(chunks))

print(chunks[0])