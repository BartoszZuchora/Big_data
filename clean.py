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

    # 1) Standaryzacja nazw kolumn
    df.columns = [c.strip() for c in df.columns]

    # 2) Konwersje typów: bool
    for c in ["adult", "video"]:
        if c in df.columns:
            # bywa, że jest jako "False"/"True" string
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
            # jeśli są braki/śmieci, wypełnij False (bezpieczny baseline)
            df[c] = df[c].fillna(False).astype(bool)

    # 3) Konwersje typów: numeryczne
    for c in ["popularity", "vote_average", "vote_count", "id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) release_date -> datetime + cechy
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

        df["release_year"] = df["release_date"].dt.year
        df["release_month"] = df["release_date"].dt.month
        df["release_quarter"] = df["release_date"].dt.quarter
        df["release_dayofweek"] = df["release_date"].dt.dayofweek

        # opcjonalnie: usuń oryginalną datę (często wygodniej zostawić cechy)
        df = df.drop(columns=["release_date"])

    # 5) genre_ids: parsowanie listy + featury
    if "genre_ids" in df.columns:

        def parse_genres(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            s = str(x).strip()
            # próba bezpiecznego parsowania "[28, 12, 878]"
            try:
                val = ast.literal_eval(s)
                if isinstance(val, list):
                    return [int(v) for v in val if str(v).isdigit() or isinstance(v, int)]
                return []
            except Exception:
                return []

        df["genre_ids"] = df["genre_ids"].apply(parse_genres)
        df["n_genres"] = df["genre_ids"].apply(len)

        # One-hot dla top N najczęstszych gatunków
        all_genres = df["genre_ids"].explode()
        all_genres = all_genres.dropna()
        if len(all_genres) > 0:
            top_genres = all_genres.value_counts().head(top_n_genres).index.tolist()
            for gid in top_genres:
                df[f"genre_{gid}"] = df["genre_ids"].apply(lambda lst, g=gid: int(g in lst))

        # usuń kolumnę listową (zostawiamy już cechy)
        df = df.drop(columns=["genre_ids"])

    # 6) Czyszczenie języka
    if "original_language" in df.columns:
        df["original_language"] = df["original_language"].astype(str).str.strip().str.lower()
        df.loc[df["original_language"].isin(["nan", "none", ""]), "original_language"] = np.nan

    # 7) Usuwanie mało-przydatnych kolumn (domyślne)
    drop_cols = []
    if drop_text_columns:
        for c in ["overview"]:
            if c in df.columns:
                drop_cols.append(c)

    # ścieżki do obrazków
    for c in ["backdrop_path", "poster_path"]:
        if c in df.columns:
            drop_cols.append(c)

    if drop_id_and_titles:
        for c in ["id", "title", "original_title"]:
            if c in df.columns:
                drop_cols.append(c)

    # usuń duplikaty w drop list
    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 8) Usuwanie wierszy z krytycznymi brakami (przykładowo)
    # Jeśli nie ma vote_count / vote_average, to często film jest mało opisany.
    critical = [c for c in ["vote_average", "vote_count", "popularity"] if c in df.columns]
    if critical:
        df = df.dropna(subset=critical)

    # 9) Proste sanity constraints (bez agresywnego wycinania)
    if "vote_average" in df.columns:
        df = df[(df["vote_average"] >= 0) & (df["vote_average"] <= 10)]

    if "vote_count" in df.columns:
        df = df[df["vote_count"] >= 0]

    if "popularity" in df.columns:
        df = df[df["popularity"] >= 0]

    # 10) Usuń duplikaty po (title, year) jeśli title został zachowany; tu zwykle title usuwamy,
    # więc robimy tylko pełne duplikaty:
    df = df.drop_duplicates()

    # 11) Zapis
    df.to_csv(output_csv_path, index=False)

    return df


# ==== UŻYCIE ====
df_clean = clean_tmdb_dataset(
    input_csv_path="data/tmdb_10000_movies.csv",
    output_csv_path="data/tmdb_10000_movies_clean.csv",
    drop_text_columns=True,
    drop_id_and_titles=True,
    top_n_genres=20,
)
print(df_clean.shape)
