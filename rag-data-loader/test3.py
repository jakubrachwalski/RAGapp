from langchain_community.vectorstores.pgvector import PGVector
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()
docs = [Document(page_content="This is a test document.")]

CONNECTION_STRING = "postgresql+psycopg2://postgres:yourpassword@localhost:5432/database164"


PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="collection164",
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True,
    use_jsonb=True
)