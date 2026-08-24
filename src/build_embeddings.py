from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CORPUS_DATA, EMBEDDINGS_DATA, MODEL_NAME


def main():
    if not CORPUS_DATA.exists():
        raise FileNotFoundError(
            f"{CORPUS_DATA} does not exist. Run: python src/prepare_data.py"
        )

    df = pd.read_csv(CORPUS_DATA).fillna("")
    texts = df["search_text"].astype(str).tolist()

    print(f"Loading embedding model:\n{MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts):,} DOAB records...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")
    EMBEDDINGS_DATA.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_DATA, embeddings)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Saved: {EMBEDDINGS_DATA}")


if __name__ == "__main__":
    main()
