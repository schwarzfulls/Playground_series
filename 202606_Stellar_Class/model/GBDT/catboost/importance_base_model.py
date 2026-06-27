# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pandas",
#     "scikit-learn",
#     "catboost",
#     "pyarrow",
#     "ruff",
#     "mypy",
# ]
# ///

"""
目的
base model の特徴量重要度を確認する
理由: 特徴量エンジニアリングで用いるため

結果
              feature  importance
8            redshift   39.079572
7                   z   12.648915
3                   u    8.850113
4                   g    7.932970
1               alpha    7.086358
9       spectral_type    6.712093
10  galaxy_population    5.496025
2               delta    4.955224
6                   i    3.903438
5                   r    3.335215
0                  id    0.000078

分かったこと
- redshift が最も重要であり, eda の pairplot の結果と一致する
  - QSO は,redshift が大きい
- 波長の中で最も予測に寄与しているのは，`z`である
  - 波長の寄与の大きい順番は, z > u > g > i > r である
- id は全く寄与しないため削除して良い

次回やること
- train data の id を削除する
  - 予測精度はほぼ同じはず
- 特徴量エンジニアリング
  - 波長の組み合わせを作成して，重要度の高い特徴量を作成する
    - 組み合わせを四則演算ごとに作成する
"""

from pathlib import Path

import pandas as pd
import polars as pl
from catboost import CatBoostClassifier

TRAIN_DF_PATH = Path("./input/")

CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]

OUTPUT_DIR = Path("./output/GBDT/catboost")


def load_df(data_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """入力データを読み込む関数

    Returns:
        pl.DataFrame: 学習データを polars で読み込んで返す
    """
    _, test_pl_df_path, train_pl_df_path = sorted(data_path.iterdir())

    train_pl_df = pl.read_csv(train_pl_df_path)
    test_pl_df = pl.read_csv(test_pl_df_path)

    return train_pl_df, test_pl_df


def preprocess_data(
    train_pl_df: pl.DataFrame, test_pl_df: pl.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train と test データの前処理を行う関数

    Args:
        train_pl_df (pl.DataFrame): train データ
        test_pl_df (pl.DataFrame): test データ

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 目的変数(class)を除外した train, 目的変数, test データ
    """
    # class 列が目的変数なので，class 列を目的変数として分割する
    x_train_pd = train_pl_df.drop("class").to_pandas(use_pyarrow_extension_array=True)
    target_train_pd = train_pl_df.select(train_pl_df.columns[-1]).to_pandas(
        use_pyarrow_extension_array=True
    )
    test_pd_df = test_pl_df.to_pandas(use_pyarrow_extension_array=True)

    return (x_train_pd, target_train_pd, test_pd_df)


def train_model(x_train_pd: pd.DataFrame, target_train_pd: pd.DataFrame) -> CatBoostClassifier:
    """catboost で学習する

    Args:
        x_train_pd (pd.DataFrame): 目的変数を除外した学習データ
        target_train_pd (pd.DataFrame): 目的変数

    Returns:
        CatBoostClassifier: 学習モデル
    """
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=96,
        verbose=100,
    )

    model.fit(x_train_pd, target_train_pd, cat_features=CATEGORICAL_COLS)

    return model


def importance_feature(model: CatBoostClassifier, x_train_pd: pd.DataFrame) -> None:
    """学習した model における特徴量重要度を算出する関数

    Args:
        model (CatBoostClassifier): catboost(base model)
        x_train_pd (pd.DataFrame): train data
    """
    # 特徴量重要度の取得
    feature_importance = model.get_feature_importance()

    importance_df = pd.DataFrame(
        {"feature": x_train_pd.columns, "importance": feature_importance}
    ).sort_values("importance", ascending=False)

    print(importance_df)


def main() -> None:

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # データの読み込み
    train_pl_df, test_pl_df = load_df(TRAIN_DF_PATH)

    # 学習データ(classなし), class データ, test データの分割する
    x_train_pd, target_train_pd, _ = preprocess_data(train_pl_df, test_pl_df)

    # train する
    model = train_model(x_train_pd, target_train_pd)

    # 特徴量重要度の確認
    importance_feature(model, x_train_pd)


if __name__ == "__main__":
    main()
