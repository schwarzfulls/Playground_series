"""base model"""

# /// script
# dependencies = [
#     "polars",
#     "scikit-learn",
#     "lightgbm",
#     "mypy",
#     "pandas",
#     "pyarrow",
#     "ruff",
#     "mypy",
# ]
# ///

from pathlib import Path

import lightgbm as lgb
import pandas as pd
import polars as pl

BASE_DIR = Path("./input/")
ALL_DATA = sorted(BASE_DIR.iterdir())
SAMPLE_SUBMISSION, TEST_DATA, TRAIN_DATA = ALL_DATA

CATEGORICAL_COLS = ["Driver", "Compound", "Race"]

OUTPUT_DIR = Path("./output/GBDT/lightbgm/")


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """使用するデータをロードする

    Returns:
        pl.DataFrame: train data
    """
    train_data = pl.read_csv(TRAIN_DATA)
    test_data = pl.read_csv(TEST_DATA)

    return (train_data, test_data)


def change_category_columns(
    train_data: pl.DataFrame, test_data: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:

    train_data = train_data.with_columns(
        [pl.col(column).cast(pl.Categorical) for column in CATEGORICAL_COLS]
    )
    test_data = test_data.with_columns(
        [pl.col(column).cast(pl.Categorical) for column in CATEGORICAL_COLS]
    )

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
    X_train_pd = train_data.drop("PitNextLap").to_pandas()
    target_train_pd = train_data.select(train_data.columns[-1]).to_pandas()

    X_test_pd = test_data.to_pandas()

    return (X_train_pd, target_train_pd, X_test_pd)


def train_model(X_train_pd: pd.DataFrame, target_train_pd: pd.Series) -> lgb:
    """light gbm による学習(base model)

    Args:
        X_train_pd (pd.DataFrame): train data
        target_train_pd (pd.Series): 正解ラベル

    Returns:
        lgb: light gbm の学習モデル
    """
    model = lgb.LGBMClassifier(
        iterations=500, learning_rate=0.05, num_leaves=31, random_seed=96, verbose=-1
    )

    model.fit(X_train_pd, target_train_pd, categorical_feature=CATEGORICAL_COLS)

    return model


def predict(model: lgb, X_test_pd: pd.DataFrame) -> pd.Series:
    """学習させたモデルを用いて予測を行う

    Args:
        model (lgb): Light GBM モデルで学習ずみのモデル
        X_test_pd (pd.DataFrame): テストデータ

    Returns:
        pd.Series: 予測結果の確率値（PitNextLap=1の確率）
    """
    return model.predict_proba(X_test_pd)[:, 1]


def create_submission_filename() -> str:
    """実行中スクリプト名を使って submission ファイル名を作成する"""

    # lightgbm_first.py
    script_name = Path(__file__).stem

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

    # CATEGORICAL_COLS を category に変換
    train_data, test_data = change_category_columns(train_data, test_data)

    X_train_pd, target_train_pd, X_test_pd = preprocess_data(train_data, test_data)

    model = train_model(X_train_pd, target_train_pd)
    pred = predict(model, X_test_pd)

    script_name = create_submission_filename()

    create_submission(pred, X_test_pd, script_name)


if __name__ == "__main__":
    main()
