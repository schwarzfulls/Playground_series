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
7           redshift   38.733559
6                  z   13.357467
2                  u    8.929202
3                  g    8.882078
0              alpha    7.087289
8      spectral_type    6.217742
9  galaxy_population    5.304282
1              delta    4.700878
5                  i    3.605596
4                  r    3.181907

id がある場合
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

予測精度: 0.94212

分かったこと
- id を削除した場合の予測精度は 0.94212 である．
  - id を削除することで，予測精度は低下するが，今回の学習データの id はただの識別子であると考えられるため，
  削除するのが分析の手法として妥当だと考える．
- id を削除しても他の特徴量重要度には大きな影響は与えない．

次回やること
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

    # id 列の削除
    x_train_pd = x_train_pd.drop("id", axis=1)

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


def predict(model: CatBoostClassifier, test_pd_df: pd.DataFrame) -> pd.Series:
    """train model を用いて predict する

    Args:
        model (CatBoostClassifier): train model
        test_pd_df (pd.DataFrame): test data

    Returns:
        pd.Series: class 名で出力するデータ
    """
    pred = model.predict(test_pd_df.drop(columns="id"))
    return pred.ravel()


def create_submission_filename() -> str:
    """実行中スクリプト名を使って submission ファイル名を作成する"""

    # catboost_first.py
    script_name = Path(__file__).stem

    return script_name


def create_submission(pred: pd.Series, X_test_pd: pd.DataFrame, filename: str) -> None:
    submission = pd.DataFrame(
        {
            "id": X_test_pd["id"],
            "class": pred,
        }
    )
    submission.to_csv(f"{OUTPUT_DIR}/submission_{filename}.csv", index=False)


def main() -> None:

    OUTPUT_DIR.mkdir(exist_ok=True)

    # データの読み込み
    train_pl_df, test_pl_df = load_df(TRAIN_DF_PATH)

    # 学習データ(classなし), class データ, test データの分割する
    x_train_pd, target_train_pd, test_pd_df = preprocess_data(train_pl_df, test_pl_df)

    # train する
    model = train_model(x_train_pd, target_train_pd)

    # 特徴量重要度の確認
    importance_feature(model, x_train_pd)

    # predict する
    pred = predict(model, test_pd_df)

    # 予測結果を保存するためにファイル名を作成する
    script_name = create_submission_filename()

    create_submission(pred, test_pd_df, script_name)


if __name__ == "__main__":
    main()
