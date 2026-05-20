"""
This script finds the best matching resumes for a job description.

What it does:
1. Loads the vector database built by resume_rag.py.
2. Reads a job description from a file.
3. Converts the job description into a vector using the same embedding model.
4. Searches the database for resume chunks that are similar to the job description.
5. Groups matching chunks by resume candidate.
6. Calculates a match score for each candidate.
7. Returns the top matches in JSON format.

Run:
    1) Activate venv (PowerShell): .\\venv\\Scripts\\Activate.ps1
    2) python job_matcher.py [job_description_file]

Example:
    python job_matcher.py jobs/jd_001_senior_data_scientist.txt

Output:
    A JSON file with top matching resumes and their match scores.
"""

import os
import json
from typing import List, Dict, Tuple
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer


def load_chroma(persist_directory: str = "./chroma_db"):
    """Load the Chroma client and the resumes collection."""
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name="resumes")
    return client, collection


def load_job_description(job_file: str) -> str:
    """Read the job description from a text file."""
    if not os.path.exists(job_file):
        raise FileNotFoundError(f"Job description file not found: {job_file}")
    with open(job_file, "r", encoding="utf-8") as f:
        return f.read()


def create_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the sentence-transformers model."""
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")
    return model


def search_resumes(
    collection,
    job_text: str,
    embed_model: SentenceTransformer,
    top_k: int = 20
) -> List[Dict]:
    """
    Search the resume collection for chunks related to the job description.
    
    Steps:
    1. Encode the job description into a vector.
    2. Query the collection for the top_k most similar resume chunks.
    3. Return the results (chunk text, metadata, similarity score).
    """
    # Convert job description text into a vector
    job_vector = embed_model.encode(job_text)
    
    # Query Chroma for top_k similar resume chunks
    results = collection.query(
        query_embeddings=[job_vector.tolist()],
        n_results=top_k
    )
    
    # Parse results into a list of dictionaries
    matches = []
    if results and results["documents"] and len(results["documents"]) > 0:
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ):
            # Convert distance to similarity score (0 to 1, where 1 is perfect match)
            # Chroma returns Euclidean distance; smaller distance = more similar
            similarity_score = 1 / (1 + dist)
            
            matches.append({
                "chunk_text": doc,
                "metadata": meta,
                "similarity_score": similarity_score
            })
    
    return matches


def aggregate_by_candidate(matches: List[Dict]) -> Dict[str, Dict]:
    """
    Group matching chunks by resume candidate and calculate aggregate scores.
    
    Steps:
    1. Group chunks by the source resume file.
    2. For each candidate, collect all matching chunks, skills, and scores.
    3. Calculate an average match score per candidate.
    4. Return a dictionary with candidate name -> aggregated data.
    """
    candidate_data = defaultdict(lambda: {
        "resume_path": None,
        "filename": None,
        "role": None,
        "seniority": None,
        "key_skills": None,
        "chunks": [],
        "similarity_scores": [],
        "sections": set(),
        "matched_keywords": set()
    })
    
    # Loop through all matching chunks
    for match in matches:
        meta = match["metadata"]
        source_file = meta.get("source_file", "unknown")
        filename = meta.get("filename", "unknown")
        
        # Use filename (without path) as the candidate key
        candidate_key = filename
        
        # Collect metadata for this candidate (same for all chunks from same resume)
        candidate_data[candidate_key]["resume_path"] = source_file
        candidate_data[candidate_key]["filename"] = filename
        candidate_data[candidate_key]["role"] = meta.get("role")
        candidate_data[candidate_key]["seniority"] = meta.get("seniority")
        candidate_data[candidate_key]["key_skills"] = meta.get("key_skills")
        
        # Collect chunk text and similarity score
        candidate_data[candidate_key]["chunks"].append(match["chunk_text"])
        candidate_data[candidate_key]["similarity_scores"].append(match["similarity_score"])
        
        # Track which sections matched (e.g., "skills", "experience")
        section = meta.get("section", "unknown")
        candidate_data[candidate_key]["sections"].add(section)
        
        # Extract keywords from the chunk (simple: split by whitespace and take long words)
        words = match["chunk_text"].split()
        long_words = [w.strip(",.;:") for w in words if len(w) > 5]
        candidate_data[candidate_key]["matched_keywords"].update(long_words[:5])
    
    return dict(candidate_data)


def calculate_match_score(candidate_info: Dict) -> float:
    """
    Calculate a final match score (0.0 to 1.0) for a candidate.
    
    Strategy:
    - Average the similarity scores of matching chunks.
    - Boost if multiple sections (skills, experience, education) matched.
    - Boost if the candidate's stated role matches the job.
    """
    if not candidate_info["similarity_scores"]:
        return 0.0
    
    # Start with average similarity score
    avg_similarity = sum(candidate_info["similarity_scores"]) / len(candidate_info["similarity_scores"])
    
    # Bonus: if multiple sections matched, boost the score (shows breadth)
    section_count = len(candidate_info["sections"])
    section_bonus = min(0.15, section_count * 0.05)
    
    final_score = min(1.0, avg_similarity + section_bonus)
    return round(final_score, 3)


def rank_candidates(candidate_data: Dict[str, Dict]) -> List[Tuple[str, Dict, float]]:
    """
    Rank candidates by their match score (highest first).
    
    Returns a list of tuples: (candidate_filename, candidate_info, match_score)
    """
    ranked = []
    for filename, info in candidate_data.items():
        score = calculate_match_score(info)
        ranked.append((filename, info, score))
    
    # Sort by score descending
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


def format_output(job_description: str, ranked_candidates: List[Tuple[str, Dict, float]], top_n: int = 5) -> Dict:
    """
    Format the final response as the assignment's JSON object.
    
    The output includes:
    - job_description: the input job description text
    - top_matches: the top N matched candidates

    Each candidate entry includes:
    - candidate_name: extracted from filename or role
    - resume_path: path to the resume file
    - match_score: final match score (0-100)
    - matched_skills: list of keywords found in matched chunks
    - relevant_excerpts: top 2-3 excerpt chunks from the resume
    - reasoning: why this candidate matched
    """
    output = []
    
    for idx, (filename, info, score) in enumerate(ranked_candidates[:top_n]):
        # Extract candidate name from filename (e.g., resume_001_aarav_mehta.txt -> Aarav Mehta)
        base = filename.replace("resume_", "").replace(".txt", "").split("_", 1)
        if len(base) > 1:
            candidate_name = base[1].replace("_", " ").title()
        else:
            candidate_name = "Unknown"
        
        # Get top 2 most relevant sections
        relevant_excerpts = info["chunks"][:3]
        
        # Convert matched keywords set to a sorted list
        matched_skills = sorted(list(info["matched_keywords"]))[:10]
        
        # Build reasoning text
        sections_str = ", ".join(sorted(info["sections"]))
        score_100 = round(score * 100, 2)
        reasoning = f"Strong match in {sections_str}. Score: {score_100}"
        if info["seniority"]:
            reasoning += f" (Seniority: {info['seniority']})"
        
        # Build output entry
        entry = {
            "candidate_name": candidate_name,
            "resume_path": info["resume_path"],
            "match_score": score_100,
            "matched_skills": matched_skills,
            "relevant_excerpts": relevant_excerpts[:2],  # Limit to 2 excerpts
            "reasoning": reasoning
        }
        
        output.append(entry)
    
    return {
        "job_description": job_description,
        "top_matches": output,
    }


def save_output(results: Dict, output_file: str = "matches.json"):
    """Save the results to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


def main():
    """Main pipeline: load DB, search for matches, rank, and output as JSON."""
    import sys
    
    # 1) Accept job file from command line or use default
    if len(sys.argv) > 1:
        job_file = sys.argv[1]
    else:
        job_file = "jobs/jd_001_senior_data_scientist.txt"
    
    print(f"Job file: {job_file}")
    
    # 2) Load job description
    job_text = load_job_description(job_file)
    print(f"Job description loaded ({len(job_text)} characters)")
    
    # 3) Load embedding model
    embed_model = create_embedding_model("all-MiniLM-L6-v2")
    
    # 4) Load Chroma collection
    print("Loading vector database...")
    client, collection = load_chroma(persist_directory="./chroma_db")
    print(f"Collection loaded with {collection.count()} chunks")
    
    # 5) Search for matching resume chunks
    print("Searching for matching resume chunks...")
    matches = search_resumes(collection, job_text, embed_model, top_k=30)
    print(f"Found {len(matches)} matching chunks")
    
    # 6) Aggregate by candidate
    print("Aggregating results by candidate...")
    candidate_data = aggregate_by_candidate(matches)
    print(f"Found {len(candidate_data)} unique candidates")
    
    # 7) Rank candidates
    ranked = rank_candidates(candidate_data)
    
    # 8) Format output (top 5)
    output = format_output(job_text, ranked, top_n=5)
    
    # 9) Print and save
    print("\n" + "=" * 60)
    print("TOP MATCHING CANDIDATES")
    print("=" * 60)
    for item in output["top_matches"]:
        print(f"\n{item['candidate_name']} (Score: {item['match_score']})")
        print(f"  Path: {item['resume_path']}")
        print(f"  Reasoning: {item['reasoning']}")
        print(f"  Skills: {', '.join(item['matched_skills'][:5])}")
    
    # Save to JSON
    save_output(output, output_file="matches.json")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
