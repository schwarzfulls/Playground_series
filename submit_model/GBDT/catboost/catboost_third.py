r"""catboost の当てはまりの良さの検証
特徴量重要度の確認

                   feature  importance
4                     Year   52.427425
7                    Stint   14.233333
8                 TyreLife    9.980997
11           LapTime_Delta    5.068622
13            RaceProgress    3.736760
3                     Race    3.520296
2                 Compound    3.432473
6                LapNumber    1.397254
14         Position_Change    1.337506
1                   Driver    1.101934
12  Cumulative_Degradation    1.023074
10             LapTime (s)    0.995160
9                 Position    0.936176
5                  PitStop    0.808989
0                       id    0.000000

-> Year の特徴量重要度が高いので，Year を drop して CV を確認する

"""

# /// script
# dependencies = [
#     "polars",
#     "scikit-learn",
#     "catboost",
#     "mypy",
#     "pandas",
#     "pyarrow",
#     "ruff",
#     "mypy",
# ]
# ///

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path("./input/")
ALL_DATA = sorted(BASE_DIR.iterdir())
SAMPLE_SUBMISSION, TEST_DATA, TRAIN_DATA = ALL_DATA

CATEGORICAL_COLS = ["Driver", "Compound", "Race"]

OUTPUT_DIR = Path("./output/GBDT/catboost")


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


def cross_validation(X_train_pd: pd.DataFrame, target_train_pd: pd.Series) -> None:
    """5 fold CV でモデルの精度を確認

    Args:
        X_train_pd (pd.DataFrame): train data
        target_train_pd (pd.Series): 正解ラベル
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=96)

    scores = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train_pd, target_train_pd)):
        X_train_fold = X_train_pd.iloc[train_idx]
        X_valid_fold = X_train_pd.iloc[valid_idx]

        y_train_fold = target_train_pd.iloc[train_idx]
        y_valid_fold = target_train_pd.iloc[valid_idx]

        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=96,
            verbose=False,
        )

        model.fit(X_train_fold, y_train_fold, cat_features=CATEGORICAL_COLS)

        pred = model.predict_proba(X_valid_fold)[:, 1]

        score = roc_auc_score(y_valid_fold, pred)

        scores.append(score)

        print(f"Fold {fold + 1}: {score:.5f}")

    print(f"Mean AUC: {np.mean(scores):.5f}")
    print(f"std AUC: {np.std(scores):.5f}")


def train_model(X_train_pd: pd.DataFrame, target_train_pd: pd.Series) -> CatBoostClassifier:
    """catboost による学習(base model)

    Args:
        X_train_pd (pd.DataFrame): train data
        target_train_pd (pd.Series): 正解ラベル

    Returns:
        CatBoostClassfier: catboost の学習モデル
    """
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=96,
        verbose=False,
    )

    model.fit(X_train_pd, target_train_pd, cat_features=CATEGORICAL_COLS)

    importance = model.get_feature_importance()
    importance_df = pd.DataFrame(
        {"feature": X_train_pd.columns, "importance": importance}
    ).sort_values("importance", ascending=False)

    print(importance_df)

    return model


def predict(model: CatBoostClassifier, X_test_pd: pd.DataFrame) -> pd.Series:
    """学習させたモデルを用いて予測を行う

    Args:
        model (CatBoostClassfier): ロジスティック回帰モデルで学習ずみのモデル
        X_test_pd (pd.DataFrame): テストデータ

    Returns:
        pd.Series: 予測結果の確率値（PitNextLap=1の確率）
    """
    return model.predict_proba(X_test_pd)[:, 1]


def create_submission_filename() -> str:
    """実行中スクリプト名を使って submission ファイル名を作成する"""

    # catboost_first.py
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

    X_train_pd, target_train_pd, X_test_pd = preprocess_data(train_data, test_data)

    # モデルの学習における特徴量重要度の確認
    model = train_model(X_train_pd, target_train_pd)
    # pred = predict(model, X_test_pd)

    # script_name = create_submission_filename()

    # create_submission(pred, X_test_pd, script_name)"""


if __name__ == "__main__":
    main()
