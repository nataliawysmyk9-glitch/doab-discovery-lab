from pathlib import Path
import sys
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DOAB_URL, RAW_DATA, CORPUS_DATA, SAMPLE_SIZE, RANDOM_SEED


ALIASES = {
    "title": [
        "dc.title",
        "title",
    ],
    "abstract": [
        "dc.description.abstract",
        "dc.description",
        "abstract",
        "description",
    ],
    "subjects": [
        "dc.subject",
        "subject",
        "subjects",
    ],
    "publisher": [
        "oapen.relation.isPublishedBy_publisher.name",
        "dc.publisher",
        "publisher",
    ],
    "uri": [
        "dc.identifier.uri",
        "handle",
        "uri",
        "url",
    ],
    "doi": [
        "oapen.identifier.doi",
        "dc.identifier.doi",
        "doi",
    ],
    "language": [
        "dc.language.iso",
        "dc.language",
        "language",
    ],
}


def download_file():
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)

    if RAW_DATA.exists():
        print(f"Raw file already exists: {RAW_DATA}")
        print("Delete it if you want to download a fresh copy.")
        return

    print(f"Downloading DOAB CSV from:\n{DOAB_URL}")
    response = requests.get(DOAB_URL, timeout=120)
    response.raise_for_status()
    RAW_DATA.write_bytes(response.content)
    print(f"Saved: {RAW_DATA}")


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        print("Standard CSV parsing failed; trying automatic delimiter detection...")
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def clean_value(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main():
    download_file()
    df = read_csv_robust(RAW_DATA)

    print(f"\nRows in source file: {len(df):,}")
    print(f"Columns in source file: {len(df.columns):,}")

    selected = {}
    for logical_name, candidates in ALIASES.items():
        found = find_column(df, candidates)
        selected[logical_name] = found
        print(f"{logical_name:10} -> {found}")

    if not selected["title"]:
        print("\nAvailable columns:")
        for column in df.columns:
            print(" -", column)
        raise RuntimeError(
            "Could not detect the title column. Add the correct name "
            "to ALIASES['title'] in prepare_data.py."
        )

    out = pd.DataFrame()

    for logical_name in ALIASES:
        source_col = selected[logical_name]
        if source_col:
            out[logical_name] = clean_value(df[source_col])
        else:
            out[logical_name] = ""

    out = out[out["title"] != ""].copy()

    # IMPORTANT:
    # All retrieval methods use exactly the same metadata representation.
    out["search_text"] = (
        "Title: " + out["title"]
        + ". Abstract: " + out["abstract"]
        + ". Subjects: " + out["subjects"]
    )

    out = out.drop_duplicates(subset=["search_text"]).reset_index(drop=True)

    if len(out) > SAMPLE_SIZE:
        out = (
            out.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
            .reset_index(drop=True)
        )

    CORPUS_DATA.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CORPUS_DATA, index=False, encoding="utf-8-sig")

    print(f"\nPrepared corpus: {len(out):,} records")
    print(f"Saved: {CORPUS_DATA}")
    print("\nExample:")
    print(
        out[["title", "publisher", "language"]]
        .head(1)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
