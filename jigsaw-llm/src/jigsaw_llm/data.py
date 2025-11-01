from pathlib import Path

import pandas as pd
import numpy as np
from scipy.special import softmax


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrameのカラム構成を自動検出して適切なexample処理を行う統合関数

    対応パターン:
    1. positive_example_1, positive_example_2, negative_example_1, negative_example_2
    2. positive_example, negative_example
    """
    df = df.copy()  # 元のDataFrameを変更しないようにコピー

    # パターン1: _1, _2のカラムが存在する場合
    pattern1_cols = ["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"]
    if all(col in df.columns for col in pattern1_cols):
        return _process_pattern1(df, pattern1_cols)

    # パターン2: 単一のexampleカラムが存在する場合
    pattern2_cols = ["positive_example", "negative_example"]
    if all(col in df.columns for col in pattern2_cols):
        return _process_pattern2(df, pattern2_cols)

    # どちらのパターンにも一致しない場合
    raise ValueError(
        "DataFrameが期待されるカラム構成と一致しません。\n"
        "パターン1: positive_example_1, positive_example_2, negative_example_1, negative_example_2\n"
        "パターン2: positive_example, negative_example",
    )


def _process_pattern1(df: pd.DataFrame, example_cols: list[str]) -> pd.DataFrame:
    """パターン1の処理: 2つのexampleを組み合わせる"""
    # 文字列のクリーニング
    for col in example_cols:
        df[col] = df[col].fillna("")
        df[col] = df[col].map(lambda x: x.strip().replace("\n\n", ""))

    # positive examplesの組み合わせ
    positive_examples = []
    for pos_1, pos_2 in zip(df["positive_example_1"], df["positive_example_2"], strict=True):
        positive_examples.append(f"```\n{pos_1}\n```\n```\n{pos_2}\n```")

    # negative examplesの組み合わせ
    negative_examples = []
    for neg_1, neg_2 in zip(df["negative_example_1"], df["negative_example_2"], strict=True):
        negative_examples.append(f"```\n{neg_1}\n```\n```\n{neg_2}\n```")

    # 新しいカラムを追加
    df["positive_examples"] = positive_examples
    df["negative_examples"] = negative_examples

    # 元のカラムを削除
    df = df.drop(columns=example_cols)
    return df


def _process_pattern2(df: pd.DataFrame, example_cols: list[str]) -> pd.DataFrame:
    """パターン2の処理: 単一のexampleを処理"""
    # 文字列のクリーニング
    for col in example_cols:
        df[col] = df[col].fillna("")
        df[col] = df[col].map(lambda x: x.strip().replace("\n\n", ""))

    # positive examplesの処理
    positive_examples = [f"```\n{pos}\n```" for pos in df["positive_example"]]
    # negative examplesの処理
    negative_examples = [f"```\n{neg}\n```" for neg in df["negative_example"]]

    # 新しいカラムを追加
    df["positive_examples"] = positive_examples
    df["negative_examples"] = negative_examples

    # 元のカラムを削除
    df = df.drop(columns=example_cols)
    return df


def read_logits_from_csv(
    df: pd.DataFrame,
    logit_csv_paths: list[str] | list[Path],
    fillna: bool = False,
) -> pd.DataFrame:
    logit_df = pd.concat(
        [pd.read_csv(Path(path)) for path in logit_csv_paths],
        ignore_index=True,
    )
    preds = softmax(logit_df[["0", "1"]].to_numpy(), axis=1)
    logit_df["0"] = preds[:, 0]
    logit_df["1"] = preds[:, 1]
    df = df.merge(
        logit_df,
        how="left",
        on="row_id",
    )
    if fillna:
        df["0"] = df.apply(lambda x: float(x["rule_violation"] == 0) if pd.isna(x["0"]) else x["0"], axis=1)
        df["1"] = df.apply(lambda x: float(x["rule_violation"] != 0) if pd.isna(x["1"]) else x["1"], axis=1)

    return df



def create_rule_violation_pairs(df: pd.DataFrame, allow_duplicates: bool) -> pd.DataFrame:
    """
    rule_violationの値(0/1)ごとにbodyをペアリングする

    Args:
        df: 'body'と'rule_violation'カラムを持つDataFrame
        allow_duplicates: 重複を許可するかどうか

    Returns:
        'body_rule_violation_0'と'body_rule_violation_1'カラムを持つDataFrame
    """
    # rule_violationごとにグループ分け
    group_0 = df[df["rule_violation"] == 0]["body"].values
    group_1 = df[df["rule_violation"] == 1]["body"].values

    len_0 = len(group_0)
    len_1 = len(group_1)

    if len_0 == 0 or len_1 == 0:
        # どちらかが空の場合は空のDataFrameを返す
        return pd.DataFrame({
            "body_rule_violation_0": [],
            "body_rule_violation_1": []
        })

    if allow_duplicates:
        # 重複あり: 多い方の数に合わせる
        n_pairs = max(len_0, len_1)

        # 多い方は重複なし、少ない方は重複ありでサンプリング
        if len_0 < len_1:
            # group_0が少ない: group_0は重複あり、group_1は重複なし
            sampled_0 = np.random.choice(group_0, size=n_pairs, replace=True)
            sampled_1 = np.random.permutation(group_1)
        elif len_1 < len_0:
            # group_1が少ない: group_1は重複あり、group_0は重複なし
            sampled_0 = np.random.permutation(group_0)
            sampled_1 = np.random.choice(group_1, size=n_pairs, replace=True)
        else:
            # 同数の場合
            sampled_0 = np.random.permutation(group_0)
            sampled_1 = np.random.permutation(group_1)
    else:
        # 重複なし: 少ない方の数に合わせる
        n_pairs = min(len_0, len_1)

        indices_0 = np.random.choice(len_0, size=n_pairs, replace=False)
        indices_1 = np.random.choice(len_1, size=n_pairs, replace=False)

        sampled_0 = group_0[indices_0]
        sampled_1 = group_1[indices_1]

    return pd.DataFrame({
        "negative_example": sampled_0,
        "positive_example": sampled_1
    })