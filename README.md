# DOAB Discovery Lab — Hybrid Search

## Goal

This version compares three retrieval strategies on the **same DOAB metadata corpus**:

1. Keyword retrieval — BM25
2. Semantic retrieval — multilingual embeddings + cosine similarity
3. Hybrid retrieval — Reciprocal Rank Fusion (RRF)

The experiment is still **retrieval-only**.

It does not use:

- RAG
- answer generation
- agents
- metadata enrichment

That makes it suitable for a workshop focused specifically on discoverability.

---

# Experiment question

Default mission:

> How does climate change influence migration and displacement?

The lab asks:

> Does hybrid retrieval combine useful lexical precision and semantic discovery better than either retrieval method alone?

---

# Why RRF?

Do not directly calculate:

```text
0.5 × BM25 + 0.5 × cosine
```

BM25 and cosine similarity use different score scales.

Instead the lab uses **Reciprocal Rank Fusion**:

```text
RRF(document)
=
1 / (60 + keyword_rank)
+
1 / (60 + semantic_rank)
```

Example:

```text
Book A
keyword rank = 2
semantic rank = 5

RRF =
1/(60+2) + 1/(60+5)
```

A document that ranks strongly in both methods gets a stronger hybrid score.

A document found strongly by only one method can still appear.

---

# Project structure

```text
discovery_lab_hybrid/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── build_embeddings.py
│   └── search_engine.py
├── data/
│   ├── raw/
│   └── processed/
└── results/
```

---

# 1. Open in VS Code

Unzip the project.

Example:

```text
C:\OAPEN\discovery-lab-hybrid
```

Open that folder in VS Code.

Then open:

```text
Terminal > New Terminal
```

---

# 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

# 3. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 4. Prepare the DOAB corpus

Run:

```powershell
python src/prepare_data.py
```

The script downloads:

```text
https://memo.doabooks.org/file/oapen/DOAB.csv
```

and creates:

```text
data\processed\doab_sample.csv
```

The default sample contains 5,000 records.

Both keyword and semantic retrieval use the same fields:

```text
title
abstract / description
subjects
```

They are combined into:

```text
search_text
```

This controls the experiment: retrieval method changes, metadata does not.

---

# 5. Inspect the corpus

Open:

```text
data\processed\doab_sample.csv
```

Look at:

```text
title
abstract
subjects
publisher
language
uri
doi
search_text
```

This is useful in the workshop because participants can see exactly what is being searched.

---

# 6. Build embeddings

Run:

```powershell
python src/build_embeddings.py
```

Output:

```text
data\processed\embeddings.npy
```

The model is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

---

# 7. Start the lab

Run:

```powershell
python -m streamlit run app.py
```

The app contains four tabs:

```text
1. Keyword
2. Semantic
3. Hybrid
4. Compare & judge
```

---

# 8. Workshop sequence

## Stage A — Keyword

Show BM25 results first.

Discuss:

- Which query terms caused retrieval?
- Is exact wording helping?
- What concepts might lexical matching miss?

## Stage B — Semantic

Reveal semantic search.

Discuss:

- Which results use different terminology?
- Are these genuinely conceptually relevant?
- Which are false positives?

## Stage C — Hybrid

Reveal hybrid results.

Explain:

```text
keyword ranking ──┐
                  ├── RRF ──> hybrid ranking
semantic ranking ─┘
```

Discuss:

- Which strong lexical results survived?
- Which semantic discoveries survived?
- Did hybrid move useful results higher?

## Stage D — Human judgement

Open:

```text
Compare & judge
```

Mark each result:

```text
Relevant? yes / no
```

The app then shows:

```text
Relevant via keyword
Relevant via semantic
Relevant via hybrid
```

and:

```text
Relevant hybrid additions vs keyword
Relevant hybrid additions vs semantic
```

---

# 9. What to measure

For a simple workshop, record:

1. relevant results in keyword top 10
2. relevant results in semantic top 10
3. relevant results in hybrid top 10
4. relevant results hybrid adds beyond keyword
5. relevant results hybrid adds beyond semantic
6. obvious false positives

Do not compare raw score values across methods.

These are different:

```text
BM25 score
cosine similarity
RRF score
```

Compare:

```text
rank
overlap
unique discoveries
human relevance
```

---

# 10. Main workshop question

The final question is:

> Does combining lexical and semantic retrieval improve discovery for this mission?

A useful outcome may be:

```text
Keyword:
precise but vocabulary-dependent

Semantic:
broader conceptual discovery but more false positives

Hybrid:
retains lexical evidence while adding semantic reach
```

Do not assume that hybrid must win.

The purpose of the lab is to test it.

---

# Architecture

```text
                    DOAB metadata
                         |
              title + abstract + subjects
                         |
                    same corpus
                         |
              +----------+----------+
              |                     |
              v                     v
         BM25 keyword          embeddings
              |                     |
              v                     v
       lexical ranking       semantic ranking
              |                     |
              +----------+----------+
                         |
                         v
                        RRF
                         |
                         v
                  HYBRID RANKING
                         |
                         v
                 HUMAN JUDGEMENT
                         |
                         v
               Does discovery improve?
```

---

# Recommended later experiments

After this hybrid retrieval lab works:

1. query expansion
2. multilingual retrieval
3. metadata-field weighting
4. larger DOAB corpus
5. evaluation with several research missions
6. only then — RAG / generated answers
