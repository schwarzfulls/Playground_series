"""base model"""

# /// script
# dependencies = [
#     "polars",
#     "scikit-learn",
#     "pandas",
#     "pyarrow",
#     "ruff",
#     "mypy",
# ]
# ///

from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path("./input/")
ALL_DATA = sorted(BASE_DIR.iterdir())
SAMPLE_SUBMISSION, TEST_DATA, TRAIN_DATA = ALL_DATA

CATEGORICAL_COLS = ["Driver", "Compound", "Race"]

OUTPUT_DIR = Path("./output/logistic/")


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """使用するデータをロードする

    Returns:
        pl.DataFrame: train data
    """
    train_data = pl.read_csv(TRAIN_DATA)
    test_data = pl.read_csv(TEST_DATA)

    return (train_data, test_data)


def preprocess_data(
    train_data: pl.DataFrame, test_data: pl.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """polars から pandas に変換して，目的変数と説明変数に分割する

    Args:
        train_data (pl.DataFrame): polars の train data
        test_data (pl.DataFrame): polars の test data

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame]: 説明変数と目的変数の pandas データと pandas に変換したテストデータ
    """
    # PitNextLap 列が目的変数であるため，最後の列を目的変数として分割する
    X_train_pd = train_data.drop("PitNextLap").to_pandas(use_pyarrow_extension_array=True)
    target_train_pd = train_data.select(train_data.columns[-1]).to_pandas(
        use_pyarrow_extension_array=True
    )

    X_test_pd = test_data.to_pandas(use_pyarrow_extension_array=True)

    return (X_train_pd, target_train_pd, X_test_pd)


def convert_one_hot_encoding(
    X_train_pd: pd.DataFrame, X_test_pd: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ "Driver", "Compound", "Race"のカテゴリ変数を one-hot encoding する

    Args:
        X_train_pd (pd.DataFrame): pandas の train data
        X_test_pd (pd.DataFrame): pandas の test data

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: one-hot encoding された pandas の train data と test data
    """
    X_train_pd = pd.get_dummies(X_train_pd, columns=CATEGORICAL_COLS)

    X_test_pd = pd.get_dummies(X_test_pd, columns=CATEGORICAL_COLS)

    X_train_pd, X_test_pd = X_train_pd.align(X_test_pd, join="left", axis=1, fill_value=0)

    return (X_train_pd, X_test_pd)


def train_model(X_train_pd: pd.DataFrame, target_train_pd: pd.Series) -> LogisticRegression:
    """ロジスティック回帰による学習

    Args:
        X_train_pd (pd.DataFrame): train data
        target_train_pd (pd.Series): 正解ラベル

    Returns:
        LogisticRegression: ロジスティック回帰の学習モデル
    """
    model = LogisticRegression()
    model.fit(X_train_pd, target_train_pd)

    return model


def predict(model: LogisticRegression, X_test_pd: pd.DataFrame) -> pd.Series:
    """学習させたモデルを用いて予測を行う

    Args:
        model (LogisticRegression): ロジスティック回帰モデルで学習ずみのモデル
        X_test_pd (pd.DataFrame): テストデータ

    Returns:
        pd.Series: 予測結果の確率値（PitNextLap=1の確率）
    """
    return model.predict_proba(X_test_pd)[:, 1]


def create_submission_filename() -> str:
    """実行中スクリプト名を使って submission ファイル名を作成する"""

    # logistic_reg_first_sub.py
    script_name = Path(__file__).stem

    # submission_logistic_reg_first_sub.csv
    return script_name


def create_submission(pred: pd.Series, X_test_pd: pd.DataFrame, filename: str) -> None:
    submission = pd.DataFrame(
        {
            "id": X_test_pd["id"],
            "PitNextLap": pred,
        }
    )
    submission.to_csv(f"{OUTPUT_DIR}/submission_{filename}.csv", index=False)


def main() -> None:

    OUTPUT_DIR.mkdir(exist_ok=True)
    train_data, test_data = load_data()

    X_train_pd, target_train_pd, X_test_pd = preprocess_data(train_data, test_data)

    pre_X_train_pd, X_test_pd = convert_one_hot_encoding(X_train_pd, X_test_pd)

    model = train_model(pre_X_train_pd, target_train_pd)
    pred = predict(model, X_test_pd)

    script_name = create_submission_filename()

    create_submission(pred, X_test_pd, script_name)


if __name__ == "__main__":
    main()
