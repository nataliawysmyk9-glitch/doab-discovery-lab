import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    CORPUS_DATA,
    EMBEDDINGS_DATA,
    MODEL_NAME,
    RRF_K,
    HYBRID_CANDIDATE_K,
)


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class DiscoveryEngine:
    df: pd.DataFrame
    embeddings: np.ndarray
    bm25: BM25Okapi
    model: SentenceTransformer

    @classmethod
    def load(cls):
        if not CORPUS_DATA.exists():
            raise FileNotFoundError(
                f"{CORPUS_DATA} not found. Run python src/prepare_data.py first."
            )

        if not EMBEDDINGS_DATA.exists():
            raise FileNotFoundError(
                f"{EMBEDDINGS_DATA} not found. Run python src/build_embeddings.py first."
            )

        df = pd.read_csv(CORPUS_DATA).fillna("")
        embeddings = np.load(EMBEDDINGS_DATA)

        if len(df) != len(embeddings):
            raise RuntimeError(
                "Corpus and embeddings have different lengths. "
                "Re-run prepare_data.py and build_embeddings.py."
            )

        tokenized_corpus = [
            tokenize(text)
            for text in df["search_text"].astype(str)
        ]

        bm25 = BM25Okapi(tokenized_corpus)
        model = SentenceTransformer(MODEL_NAME)

        return cls(
            df=df,
            embeddings=embeddings,
            bm25=bm25,
            model=model,
        )

    def _keyword_ranked_indices(self, query: str, k: int):
        scores = self.bm25.get_scores(tokenize(query))
        indices = np.argsort(scores)[::-1][:k]
        return indices, scores

    def _semantic_ranked_indices(self, query: str, k: int):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        scores = self.embeddings @ query_embedding
        indices = np.argsort(scores)[::-1][:k]
        return indices, scores

    def keyword_search(self, query: str, k: int = 10) -> pd.DataFrame:
        indices, scores = self._keyword_ranked_indices(query, k)

        result = self.df.iloc[indices].copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result.insert(1, "method", "keyword")
        result.insert(2, "score", scores[indices])
        result.insert(3, "row_id", indices)
        return result

    def semantic_search(self, query: str, k: int = 10) -> pd.DataFrame:
        indices, scores = self._semantic_ranked_indices(query, k)

        result = self.df.iloc[indices].copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result.insert(1, "method", "semantic")
        result.insert(2, "score", scores[indices])
        result.insert(3, "row_id", indices)
        return result

    def hybrid_search(self, query: str, k: int = 10) -> pd.DataFrame:
        """
        Reciprocal Rank Fusion (RRF).

        We DO NOT directly add BM25 scores and cosine-similarity scores,
        because those values live on different scales.

        For a document d:
            RRF(d) = 1 / (RRF_K + keyword_rank)
                   + 1 / (RRF_K + semantic_rank)

        Documents retrieved by only one method still receive the
        contribution from that one ranking.
        """
        candidate_k = max(HYBRID_CANDIDATE_K, k)

        kw_indices, _ = self._keyword_ranked_indices(query, candidate_k)
        sem_indices, _ = self._semantic_ranked_indices(query, candidate_k)

        keyword_rank = {
            int(row_id): rank
            for rank, row_id in enumerate(kw_indices, start=1)
        }

        semantic_rank = {
            int(row_id): rank
            for rank, row_id in enumerate(sem_indices, start=1)
        }

        all_ids = set(keyword_rank) | set(semantic_rank)

        rows = []
        for row_id in all_ids:
            kw_rank = keyword_rank.get(row_id)
            sem_rank = semantic_rank.get(row_id)

            rrf_score = 0.0

            if kw_rank is not None:
                rrf_score += 1.0 / (RRF_K + kw_rank)

            if sem_rank is not None:
                rrf_score += 1.0 / (RRF_K + sem_rank)

            rows.append(
                {
                    "row_id": row_id,
                    "keyword_rank": kw_rank,
                    "semantic_rank": sem_rank,
                    "rrf_score": rrf_score,
                }
            )

        fused = (
            pd.DataFrame(rows)
            .sort_values(
                ["rrf_score", "keyword_rank", "semantic_rank"],
                ascending=[False, True, True],
                na_position="last",
            )
            .head(k)
            .reset_index(drop=True)
        )

        records = self.df.iloc[fused["row_id"].tolist()].copy().reset_index(drop=True)

        result = pd.concat(
            [
                fused[["row_id", "keyword_rank", "semantic_rank", "rrf_score"]],
                records,
            ],
            axis=1,
        )

        result.insert(0, "rank", range(1, len(result) + 1))
        result.insert(1, "method", "hybrid")
        result.rename(columns={"rrf_score": "score"}, inplace=True)

        return result
