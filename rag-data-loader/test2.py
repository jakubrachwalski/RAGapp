import os
import psycopg

from dotenv import load_dotenv

load_dotenv()

DB_CONN_STRING = os.getenv("PGVECTOR_CONN_STRING") or "postgresql://postgres@localhost:5432/database164"
COLLECTION_NAME = "collection164"

def delete_collection_and_embeddings(conn_str: str, collection_name: str):
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            # Pobierz UUID kolekcji o danej nazwie
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            result = cur.fetchone()
            if not result:
                print(f"Collection '{collection_name}' not found, nic do usunięcia.")
                return
            collection_uuid = result[0]

            # Usuń embeddingi powiązane z kolekcją
            cur.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s", (collection_uuid,))
            print(f"Usunięto embeddingi powiązane z kolekcją '{collection_name}'.")

            # Usuń kolekcję
            cur.execute("DELETE FROM langchain_pg_collection WHERE uuid = %s", (collection_uuid,))
            print(f"Usunięto kolekcję '{collection_name}'.")

            conn.commit()

if __name__ == "__main__":
    delete_collection_and_embeddings(DB_CONN_STRING, COLLECTION_NAME)
