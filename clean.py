import ast
import numpy as np
import pandas as pd


def clean_tmdb_dataset(
    input_csv_path: str,
    output_csv_path: str = "tmdb_10000_cleaned.csv",
    drop_text_columns: bool = True,
    drop_id_and_titles: bool = True,
    top_n_genres: int = 20,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)

    df.columns = [c.strip() for c in df.columns]

    for c in ["adult", "video"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
            df[c] = df[c].fillna(False).astype(bool)

    for c in ["popularity", "vote_average", "vote_count", "id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

        df["release_year"] = df["release_date"].dt.year
        df["release_month"] = df["release_date"].dt.month
        df["release_quarter"] = df["release_date"].dt.quarter
        df["release_dayofweek"] = df["release_date"].dt.dayofweek

        df = df.drop(columns=["release_date"])

    if "genre_ids" in df.columns:

        def parse_genres(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            s = str(x).strip()
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return [int(v) for v in val if str(v).isdigit() or isinstance(v, int)]
                return []
            except Exception:
                return []

        df["genre_ids"] = df["genre_ids"].apply(parse_genres)
        df["n_genres"] = df["genre_ids"].apply(len)

        all_genres = df["genre_ids"].explode()
        all_genres = all_genres.dropna()
        if len(all_genres) > 0:
            top_genres = all_genres.value_counts().head(top_n_genres).index.tolist()
            for gid in top_genres:
                df[f"genre_{gid}"] = df["genre_ids"].apply(lambda lst, g=gid: int(g in lst))

        df = df.drop(columns=["genre_ids"])

    if "original_language" in df.columns:
        df["original_language"] = df["original_language"].astype(str).str.strip().str.lower()
        df.loc[df["original_language"].isin(["nan", "none", ""]), "original_language"] = np.nan

    drop_cols = []
    if drop_text_columns:
        for c in ["overview"]:
            if c in df.columns:
                drop_cols.append(c)

    for c in ["backdrop_path", "poster_path"]:
        if c in df.columns:
            drop_cols.append(c)

    if drop_id_and_titles:
        for c in ["id", "title", "original_title"]:
            if c in df.columns:
                drop_cols.append(c)

    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols)

    critical = [c for c in ["vote_average", "vote_count", "popularity"] if c in df.columns]
    if critical:
        df = df.dropna(subset=critical)

    if "vote_average" in df.columns:
        df = df[(df["vote_average"] >= 0) & (df["vote_average"] <= 10)]

    if "vote_count" in df.columns:
        df = df[df["vote_count"] >= 0]

    if "popularity" in df.columns:
        df = df[df["popularity"] >= 0]

    df = df.drop_duplicates()

    df.to_csv(output_csv_path, index=False)

    return df


df_clean = clean_tmdb_dataset(
    input_csv_path="data/tmdb_10000_movies.csv",
    output_csv_path="data/tmdb_10000_movies_clean.csv",
    drop_text_columns=True,
    drop_id_and_titles=True,
    top_n_genres=20,
)
print(df_clean.shape)
