import hashlib
import json
from pathlib import Path
import pandas as pd
from typing_extensions import deprecated


def make_rule_body_hash_key(df: pd.DataFrame) -> pd.DataFrame:
    def func(x: pd.Series) -> str:
        content = f"{x['rule']}||{x['body']}"
        return hashlib.sha256(content.encode()).hexdigest()[:10]

    df["rule_body_hash_key"] = df.apply(func, axis=1)
    return df


def merge_subreddit(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    _subreddit_df = pd.read_csv(csv_path)
    # dependency...
    df["body_hash"] = df["body"].map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:10])
    df = (
        df.drop(columns=["subreddit"])
        .merge(_subreddit_df, on="body_hash")
        .drop(columns=["body_hash"])
        .reset_index(drop=True)
    )
    return df

def load_rule_context(path: Path) -> dict[str, str]:
    with path.open("r") as f:
        rule_context = json.load(f)
    return rule_context


def count_label(df: pd.DataFrame) -> pd.DataFrame:
    gp = df.groupby("rule_body_hash_key", as_index=False).agg(
        rule_violation_count=("rule_violation", lambda x: x.value_counts().to_dict()),
        rule_violation_unique=("rule_violation", lambda x: x.nunique()),
        rule_violation_common=("rule_violation", lambda x: x.mode()[0]),
        is_uniform=("rule_violation", lambda x: x.value_counts().nunique() == 1),  # noqa
    )
    df = df.merge(
        gp,
        on="rule_body_hash_key",
        how="left",
    )
    return df

def replace_common_label(df: pd.DataFrame) -> pd.DataFrame:
    """ラベルがばらついているものを多数決で統一する"""
    # ラベルが同数ではないかつ、ラベルが0と1の両方が存在するもの
    cond = (~df["is_uniform"]) & (df["rule_violation_unique"] != 1)
    df.loc[cond, "rule_violation"] = df.loc[cond, "rule_violation_common"]
    return df

def drop_ambiguous_label(df: pd.DataFrame) -> pd.DataFrame:
    """ラベルが同数で0,1の頻度が同じものを削除する"""
    cond = (df["is_uniform"]) & (df["rule_violation_unique"]!=1)
    return df[~cond].reset_index(drop=True)


def _get_example_col(df: pd.DataFrame, col_name: str, is_positive: bool, suffix: str) -> pd.DataFrame:
    example_df = df[["rule", "subreddit", col_name]].rename(columns={col_name: "body"})

    example_df["rule_violation"] = 1 if is_positive else 0
    example_df["row_id"] = [f"{suffix}_{col_name}_{i}" for i in range(len(example_df))]
    return example_df


def convert_example_to_train(df: pd.DataFrame, cols: list[str], suffix: str) -> pd.DataFrame:
    examples = []
    for col in cols:
        is_positive = "positive" in col
        examples.append(_get_example_col(df, col, is_positive, suffix))

    return (pd.concat(examples).reset_index(drop=True))[["row_id", "rule", "subreddit", "body", "rule_violation"]]


@deprecated("これはもう使わない")
def train_to_without_example_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    example_df = convert_example_to_train(
        df,
        cols=cols,
        suffix="train",
    )

    df = (
        pd.concat(
            [
                df[["row_id", "rule", "subreddit", "body", "rule_violation"]],
                example_df,
            ],
        )
        .drop_duplicates(subset=["rule", "subreddit", "body"], keep="first")
        .sample(frac=1, replace=False, random_state=42)
        .reset_index(drop=True)
    )
    return df
