from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from jigsaw.data import count_label, drop_ambiguous_label, make_rule_body_hash_key, replace_common_label

app = typer.Typer()


def create_example_dataframe(_df: pd.DataFrame, sampling_ratio: float = 0.3) -> pd.DataFrame:
    """
    元のDataFrameから新しいexample付きDataFrameを作成

    Args:
        df: 元のDataFrame (row_id, body, rule_violationを持つ)
        sampling_ratio: サンプリング割合

    Returns:
        新しいDataFrame (row_id, rule, body, rule_violation, positive_example, negative_example)
    """
    df = _df.sample(frac=1.0, replace=False).reset_index(drop=True).copy()
    # positive/negativeに分割
    positive_df = df[df["rule_violation"] == 1].copy()
    negative_df = df[df["rule_violation"] == 0].copy()

    # サンプリング数を計算
    positive_sample_size = int(len(positive_df) * sampling_ratio)
    negative_sample_size = int(len(negative_df) * sampling_ratio)

    # example用をサンプリング（重複なし）
    positive_examples = positive_df.sample(n=positive_sample_size, replace=False)["body"].tolist()
    negative_examples = negative_df.sample(n=negative_sample_size, replace=False)["body"].tolist()

    # 残りのデータ（example用以外）
    remaining_positive = positive_df[~positive_df["body"].isin(positive_examples)]
    remaining_negative = negative_df[~negative_df["body"].isin(negative_examples)]

    # 残りのデータを結合
    remaining_df = pd.concat([remaining_positive, remaining_negative], ignore_index=True)

    # 新しいDataFrameの基本構造を作成
    result_df = remaining_df[["row_id", "body", "rule_violation"]].copy()

    # example列を初期化
    result_df["positive_example"] = None
    result_df["negative_example"] = None

    # exampleを順次割り当て
    positive_idx = 0
    negative_idx = 0

    for idx in range(len(result_df)):
        # positive_exampleの割り当て
        if positive_idx < len(positive_examples):
            result_df.loc[idx, "positive_example"] = positive_examples[positive_idx]
            positive_idx += 1

        # negative_exampleの割り当て
        if negative_idx < len(negative_examples):
            result_df.loc[idx, "negative_example"] = negative_examples[negative_idx]
            negative_idx += 1

    # シャッフル
    result_df = result_df.sample(frac=1.0, replace=False).reset_index(drop=True)

    return result_df


@app.command()
def rebuild_train(
    csv_path: Annotated[str, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument()],
) -> None:
    """subredditを除くとすべてのruleとexampleの組み合わせがtrainに存在しているので
    train内example列を構築しなおす
    """
    df = pd.read_csv(csv_path)[["row_id", "rule", "subreddit", "body", "rule_violation"]]
    # rule-bodyでハッシュ作成
    df = make_rule_body_hash_key(df)
    # labelをcounting
    df = count_label(df)
    # labelに統一性がないが、labelが不均衡な場合は多数派で置換
    df = replace_common_label(df)
    # ambiguousなlabelは除外
    df = drop_ambiguous_label(df)
    # rule-bodyをsubsetにして重複排除
    df = df.drop_duplicates(subset=["rule_body_hash_key"]).reset_index(drop=True)
    df = df.drop(columns=["subreddit"])

    unique_rules = df["rule"].unique()
    rebuilt_dfs = []
    for rule in unique_rules:
        rule_df = df[df["rule"] == rule].reset_index(drop=True)
        rebuilt_rule_df = create_example_dataframe(rule_df, sampling_ratio=0.3)
        rebuilt_rule_df["rule"] = rule

        # 元のrule_dfのbody, positive_example, negative_exampleの組み合わせがすべて存在していることを確認
        assert len(
            pd.concat(
                [
                    rebuilt_rule_df[rebuilt_rule_df["positive_example"].notna()][["positive_example"]].rename(
                        columns={"positive_example": "body"},
                    ),
                    rebuilt_rule_df[rebuilt_rule_df["negative_example"].notna()][["negative_example"]].rename(
                        columns={"negative_example": "body"},
                    ),
                    rebuilt_rule_df[["body"]],
                ],
            ),
        ) == len(rule_df)

        rebuilt_dfs.append(rebuilt_rule_df)

    rebuilt_df = pd.concat(rebuilt_dfs, ignore_index=True)[
        ["row_id", "rule", "body", "rule_violation", "positive_example", "negative_example"]
    ]
    rebuilt_df = rebuilt_df.sample(frac=1.0, replace=False).reset_index(drop=True)
    rebuilt_df.to_csv(output_dir.joinpath("rebuild_train.csv"), index=False)


if __name__ == "__main__":
    app()