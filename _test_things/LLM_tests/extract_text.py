import pymupdf4llm
import os
import tiktoken

data_dir = os.path.join("..", "data")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def extract_text(origin_dir, destination_dir):
    for file_name in os.listdir(origin_dir):
        content = pymupdf4llm.to_markdown(os.path.join(origin_dir, file_name))

        with open(os.path.join(destination_dir, ".".join([file_name.split(".")[0], "md"])), "w") as f:
            f.write(content)



def count_tokens(file_name):
    with open(file_name, "r") as f:
        file_content = f.read()

    encoding = tiktoken.get_encoding("o200k_base") # for GPT-4o
    num_tokens = len(encoding.encode(file_content))
    print(num_tokens)

if __name__ == "__main__":
    pdf_dir = os.path.join(data_dir, "pdf")
    md_dir = os.path.join(data_dir, "md")

    extract_text(pdf_dir, md_dir)


