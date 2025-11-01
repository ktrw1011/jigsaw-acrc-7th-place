from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from jigsaw.data import count_label, make_rule_body_hash_key, replace_common_label

app = typer.Typer()


def split(df: pd.DataFrame, fold: int, group: bool = False) -> pd.DataFrame:
    df["fold"] = -1
    if group:
        skf = StratifiedGroupKFold(n_splits=fold, shuffle=True, random_state=42)
        for i, (_, val_index) in enumerate(skf.split(df, df["rule_violation"], groups=df["rule"])):
            df.loc[val_index, "fold"] = i
    else:
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
        for i, (_, val_index) in enumerate(skf.split(df, df["rule_violation"])):
            df.loc[val_index, "fold"] = i
    return df


def select_sample(example_df: pd.DataFrame, rule: str, subreddit: str) -> str:
    cond = (example_df["rule"] == rule) & (example_df["subreddit"] == subreddit)
    if not example_df[cond].empty:
        return example_df[cond].sample(1)["body"].to_numpy()[0]
    return ""


@app.command()
def extend_df_fold(
    csv_path: Annotated[str, typer.Argument()],
    fold: Annotated[int, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()],
    group: Annotated[bool, typer.Option("-g", flag_value=True)] = False,
) -> None:
    """example列なしでデータフレームを作成してfold分割する"""
    output_dir.mkdir(parents=True, exist_ok=True)
    # ここのsubredditは正しくない
    df = pd.read_csv(csv_path)[["row_id", "rule", "subreddit", "body", "rule_violation"]]
    # rule-bodyでハッシュ作成
    df = make_rule_body_hash_key(df)
    # labelをcounting
    df = count_label(df)
    # labelに統一性がないが、labelが不均衡な場合は多数派で置換
    df = replace_common_label(df)
    # rule-bodyをsubsetにして重複排除
    df = df.drop_duplicates(subset=["rule_body_hash_key"]).reset_index(drop=True)
    prefix = "sgkf" if group else "skf"
    df = split(df, fold, group)
    df.to_csv(output_dir.joinpath(f"{prefix}_wo_example_{fold}folds.csv"), index=False)


@app.command()
def extend_with_example_df_fold(
    csv_path: Annotated[str, typer.Argument()],
    fold: Annotated[int, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()],
    group: Annotated[bool, typer.Option("-g", flag_value=True)] = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(csv_path)

    postive_example_df = (
        train_df[["rule", "subreddit", "positive_example_2"]]
        .rename(columns={"positive_example_2": "body"})
        .drop_duplicates(subset=["body"])
    )
    negative_example_df = (
        train_df[["rule", "subreddit", "negative_example_2"]]
        .rename(columns={"negative_example_2": "body"})
        .drop_duplicates(subset=["body"])
    )

    extended_df = []
    for col in ["positive_example_1", "negative_example_1"]:
        _df = train_df[["row_id", "subreddit", "rule", col]].copy()
        _df = _df.rename(columns={col: "body"})
        _df["original"] = False
        _df["rule_violation"] = 1 if "positive" in col else 0
        _df["row_id"] = _df["row_id"].astype(str) + "_" + col
        extended_df.append(_df)

    extended_df = pd.concat(extended_df, ignore_index=True)
    extended_df = extended_df.drop_duplicates(subset=["subreddit", "rule", "body"], keep="first")

    original_df = train_df[["row_id", "subreddit", "rule", "body", "rule_violation"]].copy()
    original_df["original"] = True
    df = pd.concat([extended_df, original_df]).reset_index(drop=True)
    df = df.sample(frac=1, replace=False).reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        # positive exampleとnegative exampleのランダムサンプリングして追加する
        pos_sample = select_sample(postive_example_df, row["rule"], row["subreddit"])
        neg_sample = select_sample(negative_example_df, row["rule"], row["subreddit"])
        row["positive_example"] = pos_sample
        row["negative_example"] = neg_sample
        rows.append(row)

        # addしたexampleはexample_dfから削除する
        if pos_sample:
            cond_pos = (
                (postive_example_df["rule"] == row["rule"])
                & (postive_example_df["subreddit"] == row["subreddit"])
                & (postive_example_df["body"] == pos_sample)
            )
            postive_example_df = postive_example_df[~cond_pos].reset_index(drop=True)

        # ネガティブサンプルが見つかった場合、該当するすべての行を削除
        if neg_sample:
            cond_neg = (
                (negative_example_df["rule"] == row["rule"])
                & (negative_example_df["subreddit"] == row["subreddit"])
                & (negative_example_df["body"] == neg_sample)
            )
            negative_example_df = negative_example_df[~cond_neg].reset_index(drop=True)

    df = pd.DataFrame(rows)
    prefix = "sgkf" if group else "skf"
    df = split(df, fold, group)

    df.to_csv(output_dir.joinpath(f"extend_example_{prefix}_{fold}folds.csv"), index=False)


@app.command()
def simple_fold(
    csv_path: Annotated[str, typer.Argument()],
    fold: Annotated[int, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()],
    group: Annotated[bool, typer.Option("-g", flag_value=True)] = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    prefix = "sgkf" if group else "skf"
    df = split(df, fold, group)
    df.to_csv(output_dir.joinpath(f"{prefix}_{fold}folds.csv"), index=False)


if __name__ == "__main__":
    app()
