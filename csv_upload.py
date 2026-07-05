"""Helpers for CSV batch sentiment analysis."""

from __future__ import annotations

import pandas as pd


TEXT_COLUMN_ALIASES = (
    "review",
    "text",
    "tweet text",
    "tweet_text",
    "tweet",
    "content",
    "comment",
    "message",
    "feedback",
    "body",
)


def normalize_column_name(name: object) -> str:
    return " ".join(str(name).replace("_", " ").replace("-", " ").lower().split())


def find_text_column(dataframe: pd.DataFrame) -> str | None:
    normalized_columns = {
        normalize_column_name(column): str(column) for column in dataframe.columns
    }
    for alias in TEXT_COLUMN_ALIASES:
        column = normalized_columns.get(normalize_column_name(alias))
        if column is not None:
            return column

    string_columns = [
        str(column)
        for column in dataframe.columns
        if pd.api.types.is_string_dtype(dataframe[column])
    ]
    if len(string_columns) == 1:
        return string_columns[0]
    return None


def extract_text_rows(dataframe: pd.DataFrame) -> tuple[str | None, pd.Series]:
    text_column = find_text_column(dataframe)
    if text_column is None:
        return None, pd.Series(dtype=str)

    text_rows = dataframe[text_column].dropna().astype(str).str.strip()
    text_rows = text_rows[text_rows != ""]
    return text_column, text_rows


def build_batch_predictions(dataframe: pd.DataFrame, classifier) -> pd.DataFrame:
    text_column, text_rows = extract_text_rows(dataframe)
    if text_column is None or text_rows.empty:
        return pd.DataFrame()

    predictions = classifier(text_rows.tolist())
    results = dataframe.loc[text_rows.index].copy()
    results["source_text_column"] = text_column
    results["predicted_sentiment"] = [prediction["label"] for prediction in predictions]
    results["confidence"] = [prediction["score"] for prediction in predictions]
    return results
