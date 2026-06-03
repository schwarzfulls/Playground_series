r"""base model

既に決まっていること
- catboost
- id は削除
- Year は CV が若干高いため残す
- LightGBM とのアンサンブルをしても精度は上昇しない

今回やること
- catboost のハイパラのチューニングを行う
  1. 深い木
  2. さらに深い木
  3. L2正則化

このスクリプトは 3 を行う

結果
Fold 1: 0.94593
Fold 2: 0.94499
Fold 3: 0.94668
Fold 4: 0.94592
Fold 5: 0.94619
Mean AUC: 0.94594
std AUC: 0.00055

結論
L2正則化は効果がない

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
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
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

    # id は特徴量重要度が 0 のため  train と test から削除する
    X_test_id = pd.DataFrame({"id": X_test_pd["id"]})
    X_train_pd = X_train_pd.drop(columns=["id"])
    X_test_pd = X_test_pd.drop(columns=["id"])

    return (X_train_pd, target_train_pd, X_test_pd, X_test_id)


def train_model(X_train_pd: pd.DataFrame, target_train_pd: pd.Series) -> CatBoostClassifier:
    """catboost による学習(base model)

    Args:
        X_train_pd (pd.DataFrame): train data
        target_train_pd (pd.Series): 正解ラベル

    Returns:
        CatBoostClassfier: catboost の学習モデル
    """
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=96,
        verbose=False,
    )

    model.fit(X_train_pd, target_train_pd, cat_features=CATEGORICAL_COLS)

    return model


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

        model = train_model(X_train_fold, y_train_fold)

        pred = model.predict_proba(X_valid_fold)[:, 1]

        score = roc_auc_score(y_valid_fold, pred)

        scores.append(score)

        print(f"Fold {fold + 1}: {score:.5f}")

    print(f"Mean AUC: {np.mean(scores):.5f}")
    print(f"std AUC: {np.std(scores):.5f}")


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

    X_train_pd, target_train_pd, X_test_pd, X_test_id = preprocess_data(train_data, test_data)

    cross_validation(X_train_pd, target_train_pd)

    model = train_model(X_train_pd, target_train_pd)
    pred = predict(model, X_test_pd)

    script_name = create_submission_filename()

    create_submission(pred, X_test_id, script_name)


if __name__ == "__main__":
    main()
