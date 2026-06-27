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


def predict(model: CatBoostClassifier, test_pd_df: pd.DataFrame) -> pd.Series:
    """train model を用いて predict する

    Args:
        model (CatBoostClassifier): train model
        test_pd_df (pd.DataFrame): test data

    Returns:
        pd.Series: class 名で出力するデータ
    """
    pred = model.predict(test_pd_df)
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

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # データの読み込み
    train_pl_df, test_pl_df = load_df(TRAIN_DF_PATH)

    # 学習データ(classなし), class データ, test データの分割する
    x_train_pd, target_train_pd, test_pd_df = preprocess_data(train_pl_df, test_pl_df)

    # train する
    model = train_model(x_train_pd, target_train_pd)

    # predict する
    pred = predict(model, test_pd_df)

    # 予測結果を保存するためにファイル名を作成する
    script_name = create_submission_filename()

    create_submission(pred, test_pd_df, script_name)


if __name__ == "__main__":
    main()
