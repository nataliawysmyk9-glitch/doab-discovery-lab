import pandas as pd
import streamlit as st

from config import (
    MISSION_QUERY,
    TOP_K,
    JUDGEMENTS_DATA,
    MODEL_NAME,
    RRF_K,
    HYBRID_CANDIDATE_K,
)
from src.search_engine import DiscoveryEngine, tokenize


st.set_page_config(
    page_title="DOAB Discovery Lab — Hybrid",
    layout="wide",
)


@st.cache_resource
def load_engine():
    return DiscoveryEngine.load()


def short_text(value: str, limit: int = 350) -> str:
    value = str(value or "")
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "…"


def display_results(df: pd.DataFrame, title: str, score_label: str, query_tokens=None):
    st.subheader(title)

    for _, row in df.iterrows():
        with st.container(border=True):
            st.markdown(f"**#{int(row['rank'])} — {row['title']}**")

            meta = []

            if row.get("publisher"):
                meta.append(str(row["publisher"]))

            if row.get("language"):
                meta.append(f"language: {row['language']}")

            if meta:
                st.caption(" | ".join(meta))

            if row.get("abstract"):
                st.write(short_text(row.get("abstract", "")))

            # Transparency: show which query tokens literally occur
            # in the same metadata text used by BM25.
            if query_tokens:
                searchable = str(row.get("search_text", "")).lower()
                matched = [token for token in query_tokens if token.lower() in searchable]

                if matched:
                    st.caption(
                        "Exact query terms found in metadata: "
                        + " | ".join(matched)
                    )
                else:
                    st.caption("Exact query terms found in metadata: none")

            cols = st.columns([1, 1, 1, 3])

            cols[0].metric(score_label, f"{float(row['score']):.4f}")

            if "keyword_rank" in row.index and pd.notna(row.get("keyword_rank")):
                cols[1].metric("KW rank", int(row["keyword_rank"]))

            if "semantic_rank" in row.index and pd.notna(row.get("semantic_rank")):
                cols[2].metric("SEM rank", int(row["semantic_rank"]))

            if row.get("uri"):
                cols[3].write(row["uri"])


st.title("DOAB Discovery Lab — Hybrid Retrieval")
st.write(
    "Compare three retrieval strategies on the same DOAB corpus: "
    "**keyword**, **semantic**, and **hybrid**."
)

with st.expander("What is happening under the hood?", expanded=False):
    st.markdown(
        f"""
### 1. Keyword retrieval
The query is tokenized and ranked with **BM25**.

### 2. Semantic retrieval
DOAB metadata records and the query are embedded with:

`{MODEL_NAME}`

The system ranks documents by cosine similarity.

### 3. Hybrid retrieval
Hybrid search combines the two rankings using **Reciprocal Rank Fusion (RRF)**.

For a document:

`RRF = 1 / ({RRF_K} + keyword rank) + 1 / ({RRF_K} + semantic rank)`

The hybrid method first looks at the top `{HYBRID_CANDIDATE_K}` candidates
from each retrieval route.

### Why use RRF?
BM25 scores and cosine-similarity scores are not directly comparable.
RRF combines **rank positions**, not raw scores.

There is **no generative AI or RAG** in this experiment.
"""
    )

try:
    engine = load_engine()
except Exception as exc:
    st.error(str(exc))
    st.stop()

query = st.text_input(
    "Research question",
    value=MISSION_QUERY,
)

# Show exactly what the lexical/BM25 route receives.
query_tokens = tokenize(query)

st.markdown("**BM25 query tokens used:**")
if query_tokens:
    st.code(" | ".join(query_tokens))
else:
    st.code("(none)")

st.caption(
    "These are the tokens passed to BM25 after lower-casing and tokenization. "
    "No stop-word removal is applied in this version."
)

top_k = st.slider(
    "Results per method",
    min_value=5,
    max_value=20,
    value=TOP_K,
)

if query.strip():
    keyword = engine.keyword_search(query, top_k)
    semantic = engine.semantic_search(query, top_k)
    hybrid = engine.hybrid_search(query, top_k)

    tabs = st.tabs(
        [
            "1. Keyword",
            "2. Semantic",
            "3. Hybrid",
            "4. Compare & judge",
        ]
    )

    with tabs[0]:
        display_results(
            keyword,
            "Keyword / BM25 results",
            "BM25 score",
            query_tokens=query_tokens,
        )

    with tabs[1]:
        display_results(
            semantic,
            "Semantic results",
            "Cosine",
            query_tokens=query_tokens,
        )

    with tabs[2]:
        display_results(
            hybrid,
            "Hybrid / RRF results",
            "RRF score",
            query_tokens=query_tokens,
        )

        st.info(
            "The RRF score is only used to rank the hybrid result set. "
            "Do not compare its numeric value with BM25 or cosine scores."
        )

    with tabs[3]:
        keyword_ids = keyword["row_id"].astype(int).tolist()
        semantic_ids = semantic["row_id"].astype(int).tolist()
        hybrid_ids = hybrid["row_id"].astype(int).tolist()

        kw_set = set(keyword_ids)
        sem_set = set(semantic_ids)
        hyb_set = set(hybrid_ids)

        keyword_rank = dict(zip(keyword_ids, keyword["rank"]))
        semantic_rank = dict(zip(semantic_ids, semantic["rank"]))
        hybrid_rank = dict(zip(hybrid_ids, hybrid["rank"]))

        union_ids = list(
            dict.fromkeys(
                keyword_ids
                + semantic_ids
                + hybrid_ids
            )
        )

        rows = []

        for row_id in union_ids:
            source = engine.df.iloc[int(row_id)]

            methods = []
            if row_id in kw_set:
                methods.append("keyword")
            if row_id in sem_set:
                methods.append("semantic")
            if row_id in hyb_set:
                methods.append("hybrid")

            rows.append(
                {
                    "row_id": int(row_id),
                    "title": source["title"],
                    "publisher": source.get("publisher", ""),
                    "keyword_rank": keyword_rank.get(row_id, None),
                    "semantic_rank": semantic_rank.get(row_id, None),
                    "hybrid_rank": hybrid_rank.get(row_id, None),
                    "found_by": ", ".join(methods),
                    "relevant": False,
                    "notes": "",
                }
            )

        judgement_df = pd.DataFrame(rows)

        st.markdown("## Compare & judge")
        st.caption(
            "Judge relevance first. The summary below updates from your human decisions."
        )

        edited = st.data_editor(
            judgement_df,
            hide_index=True,
            disabled=[
                "row_id",
                "title",
                "publisher",
                "keyword_rank",
                "semantic_rank",
                "hybrid_rank",
                "found_by",
            ],
            column_config={
                "relevant": st.column_config.CheckboxColumn("Relevant?"),
                "keyword_rank": st.column_config.NumberColumn("Keyword rank", format="#%d"),
                "semantic_rank": st.column_config.NumberColumn("Semantic rank", format="#%d"),
                "hybrid_rank": st.column_config.NumberColumn("Hybrid rank", format="#%d"),
            },
            use_container_width=True,
            key="judgement_editor",
        )

        relevant_ids = set(
            edited.loc[
                edited["relevant"] == True,
                "row_id"
            ].astype(int)
        )

        kw_rel = len(relevant_ids & kw_set)
        sem_rel = len(relevant_ids & sem_set)
        hyb_rel = len(relevant_ids & hyb_set)

        st.markdown("### Relevance summary")

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "Keyword",
            f"{kw_rel}/{len(keyword_ids)} relevant",
            f"{(kw_rel/len(keyword_ids)*100):.0f}%" if keyword_ids else "0%"
        )

        s2.metric(
            "Semantic",
            f"{sem_rel}/{len(semantic_ids)} relevant",
            f"{(sem_rel/len(semantic_ids)*100):.0f}%" if semantic_ids else "0%"
        )

        s3.metric(
            "Hybrid",
            f"{hyb_rel}/{len(hybrid_ids)} relevant",
            f"{(hyb_rel/len(hybrid_ids)*100):.0f}%" if hybrid_ids else "0%"
        )

        hybrid_added_vs_keyword = relevant_ids & (hyb_set - kw_set)
        hybrid_added_vs_semantic = relevant_ids & (hyb_set - sem_set)

        g1, g2 = st.columns(2)

        g1.metric(
            "Relevant hybrid additions vs keyword",
            len(hybrid_added_vs_keyword),
        )

        g2.metric(
            "Relevant hybrid additions vs semantic",
            len(hybrid_added_vs_semantic),
        )

        st.markdown("### Rank comparison")

        sort_option = st.selectbox(
            "Sort comparison by",
            [
                "Hybrid rank",
                "Keyword rank",
                "Semantic rank",
                "Relevant first",
                "Hybrid additions",
            ],
        )

        comparison = edited.copy()

        # Friendly display values.
        comparison["Keyword"] = comparison["keyword_rank"].apply(
            lambda x: f"#{int(x)}" if pd.notna(x) else "—"
        )
        comparison["Semantic"] = comparison["semantic_rank"].apply(
            lambda x: f"#{int(x)}" if pd.notna(x) else "—"
        )
        comparison["Hybrid"] = comparison["hybrid_rank"].apply(
            lambda x: f"#{int(x)}" if pd.notna(x) else "—"
        )
        comparison["Relevant"] = comparison["relevant"].apply(
            lambda x: "✓" if bool(x) else ""
        )

        if sort_option == "Hybrid rank":
            comparison = comparison.sort_values(
                ["hybrid_rank", "semantic_rank", "keyword_rank"],
                na_position="last"
            )
        elif sort_option == "Keyword rank":
            comparison = comparison.sort_values(
                ["keyword_rank", "hybrid_rank", "semantic_rank"],
                na_position="last"
            )
        elif sort_option == "Semantic rank":
            comparison = comparison.sort_values(
                ["semantic_rank", "hybrid_rank", "keyword_rank"],
                na_position="last"
            )
        elif sort_option == "Relevant first":
            comparison = comparison.sort_values(
                ["relevant", "hybrid_rank"],
                ascending=[False, True],
                na_position="last"
            )
        elif sort_option == "Hybrid additions":
            comparison["_hybrid_addition"] = comparison["row_id"].apply(
                lambda x: int(x) in (hyb_set - kw_set)
            )
            comparison = comparison.sort_values(
                ["_hybrid_addition", "hybrid_rank"],
                ascending=[False, True],
                na_position="last"
            )

        st.dataframe(
            comparison[
                [
                    "title",
                    "publisher",
                    "Keyword",
                    "Semantic",
                    "Hybrid",
                    "Relevant",
                ]
            ].rename(
                columns={
                    "title": "Title",
                    "publisher": "Publisher",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("### What did hybrid add?")

        hybrid_addition_rows = edited[
            edited["row_id"].astype(int).isin(hyb_set - kw_set)
        ].copy()

        if len(hybrid_addition_rows):
            for _, row in hybrid_addition_rows.sort_values(
                "hybrid_rank",
                na_position="last"
            ).iterrows():
                symbol = "✓" if bool(row["relevant"]) else "○"
                sem_rank = (
                    f"Semantic #{int(row['semantic_rank'])}"
                    if pd.notna(row["semantic_rank"])
                    else "Not in semantic top results"
                )
                hyb_rank = (
                    f"Hybrid #{int(row['hybrid_rank'])}"
                    if pd.notna(row["hybrid_rank"])
                    else ""
                )

                st.markdown(
                    f"**{symbol} {row['title']}**  \n"
                    f"{sem_rank} → {hyb_rank}"
                )
        else:
            st.info(
                "For this query, the Hybrid top results contain no titles "
                "outside the Keyword top results."
            )

        st.markdown("### Relevant lexical results preserved by hybrid")

        preserved_ids = relevant_ids & kw_set & hyb_set

        if preserved_ids:
            preserved = edited[
                edited["row_id"].astype(int).isin(preserved_ids)
            ].sort_values("hybrid_rank", na_position="last")

            for _, row in preserved.iterrows():
                st.markdown(
                    f"**✓ {row['title']}**  \n"
                    f"Keyword #{int(row['keyword_rank'])} → "
                    f"Hybrid #{int(row['hybrid_rank'])}"
                )
        else:
            st.caption(
                "Mark relevant results above to see which lexical results "
                "were preserved by the hybrid ranking."
            )

        st.markdown("### Interpretation")

        if not relevant_ids:
            st.info(
                "Mark titles as relevant or not relevant above. "
                "The interpretation will then summarize the experiment."
            )
        else:
            best = max(
                [("Keyword", kw_rel), ("Semantic", sem_rel), ("Hybrid", hyb_rel)],
                key=lambda x: x[1]
            )

            if best[0] == "Hybrid" and hyb_rel > max(kw_rel, sem_rel):
                st.success(
                    f"For this query, Hybrid has the highest number of "
                    f"human-judged relevant results ({hyb_rel}/{len(hybrid_ids)}). "
                    "It appears to combine lexical evidence with semantic reach."
                )
            elif hyb_rel == max(kw_rel, sem_rel):
                st.info(
                    "For this query, Hybrid matches the best-performing retrieval "
                    "method on human-judged relevance. Inspect the rank comparison "
                    "to see whether it still improves ordering or preserves useful results."
                )
            else:
                st.warning(
                    "For this query, Hybrid does not produce the highest relevance count. "
                    "That is still a useful experimental result and should be discussed."
                )

        st.caption(
            "Interpretation applies only to this query, corpus and configuration. "
            "Do not compare raw BM25, cosine and RRF score values directly."
        )

        if st.button("Save judgement"):
            JUDGEMENTS_DATA.parent.mkdir(parents=True, exist_ok=True)

            output = edited.copy()
            output.insert(0, "query", query)

            output.to_csv(
                JUDGEMENTS_DATA,
                index=False,
                encoding="utf-8-sig",
            )

            st.success(f"Saved to {JUDGEMENTS_DATA}")
