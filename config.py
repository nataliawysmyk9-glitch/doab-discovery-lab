from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DOAB_URL = "https://memo.doabooks.org/file/oapen/DOAB.csv"

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "DOAB.csv"
CORPUS_DATA = PROJECT_ROOT / "data" / "processed" / "doab_sample.csv"
EMBEDDINGS_DATA = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
JUDGEMENTS_DATA = PROJECT_ROOT / "results" / "judgements.csv"

SAMPLE_SIZE = 5000
RANDOM_SEED = 42
TOP_K = 10

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MISSION_QUERY = "How does climate change influence migration and displacement?"

# Reciprocal Rank Fusion constant.
# A larger value makes differences between nearby ranks less extreme.
RRF_K = 60

# Hybrid search first retrieves a wider candidate pool from both systems.
HYBRID_CANDIDATE_K = 50
