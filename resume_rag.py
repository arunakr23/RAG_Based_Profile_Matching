"""
This script builds the resume vector database for the project.

What it does:
1. Reads `data_index.csv` for file names and metadata.
2. Loads each resume text file from `resumes/`.
3. Splits resumes into sections like Skills, Experience, and Education.
4. Breaks long sections into smaller chunks.
5. Creates embeddings with a local SentenceTransformer model.
6. Saves the chunk vectors and metadata in ChromaDB (`./chroma_db`).

Run:
    1) Activate venv (PowerShell): .\\venv\\Scripts\\Activate.ps1
    2) python resume_rag.py

The first run downloads the embedding model, so internet access is needed once.
`data_index.csv` should already exist in the project folder.
"""

import os
import uuid
from typing import List, Tuple, Dict

import pandas as pd

# Local embedding model
from sentence_transformers import SentenceTransformer

# ChromaDB for vector storage
import chromadb


def read_index(csv_path: str = "data_index.csv") -> pd.DataFrame:
    """Read the CSV file that contains resume paths and metadata.

    Expected columns: filename, path, role, seniority, key_skills, expected_relevance
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Index file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def load_text(path: str) -> str:
    """Read a text file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Split resume text into sections using simple heading detection."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    headings = [
        "summary",
        "skills",
        "experience",
        "work experience",
        "projects",
        "education",
        "certifications",
        "publications",
    ]

    sections: Dict[str, List[str]] = {}
    current = "document"
    sections[current] = []

    for line in lines:
        if not line.strip():
            continue
        low = line.lower().strip().rstrip(":")
        if any(low.startswith(h) for h in headings):
            current = low
            if current not in sections:
                sections[current] = []
            continue
        sections[current].append(line)

    result: List[Tuple[str, str]] = []
    for sec_name, sec_lines in sections.items():
        text_block = "\n".join(sec_lines).strip()
        if text_block:
            result.append((sec_name, text_block))
    return result


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Split a text into overlapping character chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def create_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the sentence-transformers model used for embeddings."""
    print(f"Loading sentence-transformers model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")
    return model


def init_chroma(persist_directory: str = "./chroma_db"):
    """Initialize Chroma and create the `resumes` collection."""
    client = chromadb.PersistentClient(path=persist_directory)

    # Delete the old collection if it already exists
    existing = [c.name for c in client.list_collections()]
    if "resumes" in existing:
        try:
            client.delete_collection(name="resumes")
        except Exception:
            pass

    collection = client.create_collection(name="resumes")
    return client, collection


def ingest_all(df: pd.DataFrame, embed_model: SentenceTransformer, collection) -> int:
    """Add all resume chunks into the Chroma collection."""
    total_chunks = 0

    for _, row in df.iterrows():
        # Pick the file path from the CSV row
        if "path" in row and pd.notna(row["path"]):
            path = row["path"]
        else:
            path = os.path.join("resumes", row["filename"])

        if not os.path.exists(path):
            print(f"Warning: resume not found, skipping: {path}")
            continue

        text = load_text(path)

        # Split the resume into sections first
        sections = split_into_sections(text)

        for sec_name, sec_text in sections:
            chunks = chunk_text(sec_text)
            for idx, chunk in enumerate(chunks):
                # Build metadata for this chunk
                metadata = {
                    "source_file": path,
                    "filename": os.path.basename(path),
                    "section": sec_name,
                    "chunk_index": idx,
                }
                # Add extra columns from the CSV file
                for k in ["role", "seniority", "key_skills", "expected_relevance"]:
                    if k in row and pd.notna(row[k]):
                        metadata[k] = row[k]

                # Create an ID for this chunk
                chunk_id = f"{os.path.basename(path)}::chunk::{idx}::{uuid.uuid4().hex[:8]}"

                # Convert text into an embedding vector
                vector = embed_model.encode(chunk)

                # Store the chunk inside Chroma
                collection.add(ids=[chunk_id], documents=[chunk], metadatas=[metadata], embeddings=[vector.tolist()])

                total_chunks += 1

        print(f"Indexed: {path} (chunks so far: {total_chunks})")

    return total_chunks


def main():
    # 1) Read the CSV file
    df = read_index("data_index.csv")

    # 2) Load the embedding model
    embed_model = create_embedding_model("all-MiniLM-L6-v2")

    # 3) Initialize Chroma
    client, collection = init_chroma(persist_directory="./chroma_db")

    # 4) Ingest the resume chunks
    total = ingest_all(df, embed_model, collection)

    # 5) Save the database
    try:
        client.persist()
    except Exception:
        pass

    print(f"\nIngestion complete. Total chunks indexed: {total}")


if __name__ == "__main__":
    main()
