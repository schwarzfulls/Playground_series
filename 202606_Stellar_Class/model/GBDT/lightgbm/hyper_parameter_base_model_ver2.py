# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pandas",
#     "scikit-learn",
#     "lightgbm",
#     "pyarrow",
#     "ruff",
#     "mypy",
# ]
# ///

"""
目的
base model(LightGBM)のハイパーパラメータを決める(2回目)

予測精度Score: 0.95690

--------------------------------------------------
CV Mean : 0.968308
CV Std  : 0.000476
--------------------------------------------------

特徴量重要度
Best iteration : 577
         feature  importance
0          alpha       50418
1          delta       42816
16   alpha_delta       32753
7       redshift       27820
10           r_i       26862
6              z       25112
11           i_z       24716
9            g_r       24494
8            u_g       24397
2              u       24080
3              g       21547
15  redshift_i_z       19757
14  redshift_r_i       19588
13  redshift_g_r       19541
12  redshift_u_g       19191
4              r       18665
5              i       17917

分かったこと
- 予測精度が低下する

次回やること
- 特徴量エンジニアリング

"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

TRAIN_DF_PATH = Path("./input/")
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]

OUTPUT_DIR = Path("./output/GBDT/lightgbm")

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    # Boosting
    "n_estimators": 5000,
    "learning_rate": 0.02,
    # Tree
    "max_depth": -1,
    "num_leaves": 255,
    "min_child_samples": 40,
    # Sampling
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    # Regularization
    "lambda_l1": 0.5,
    "lambda_l2": 1.0,
    "random_state": 96,
}


def load_df(data_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """入力データを読み込む関数

    Returns:
        pl.DataFrame: 学習データを polars で読み込んで返す
    """
    _, test_pl_df_path, train_pl_df_path = sorted(data_path.iterdir())

    train_pl_df = pl.read_csv(train_pl_df_path)
    test_pl_df = pl.read_csv(test_pl_df_path)

    return train_pl_df, test_pl_df


def feat_eng_spectrum(
    train_pl_df: pl.DataFrame, test_pl_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """5つの波長をそれぞれ差をとって新しい特徴量を作成する関数

    Args:
        train_pl_df (pl.DataFrame): train data
        test_pl_df (pl.DataFrame): test data

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: 新しい特徴量を追加した train, test data
    """
    df_list = [train_pl_df, test_pl_df]
    calc_df_list = []
    for df in df_list:
        calc_df_list.append(
            df.with_columns(
                [
                    (pl.col("u") - pl.col("g")).alias("u_g"),
                    (pl.col("g") - pl.col("r")).alias("g_r"),
                    (pl.col("r") - pl.col("i")).alias("r_i"),
                    (pl.col("i") - pl.col("z")).alias("i_z"),
                ]
            )
        )

    return calc_df_list[0], calc_df_list[1]


def feat_eng_spectrum_redshift(
    train_pl_df: pl.DataFrame, test_pl_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """redshift を波長の積を新しい特徴量として追加する関数

    Args:
        train_pl_df (pl.DataFrame): train data
        test_pl_df (pl.DataFrame): test data

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: redshift * spectrum を追加した train, test データ
    """
    df_list = [train_pl_df, test_pl_df]
    calc_df_list = []

    for df in df_list:
        df = df.with_columns((pl.col("redshift") * pl.col("u_g")).alias("redshift_u_g"))
        df = df.with_columns((pl.col("redshift") * pl.col("g_r")).alias("redshift_g_r"))
        df = df.with_columns((pl.col("redshift") * pl.col("r_i")).alias("redshift_r_i"))
        df = df.with_columns((pl.col("redshift") * pl.col("i_z")).alias("redshift_i_z"))

        calc_df_list.append(df)

    return calc_df_list[0], calc_df_list[1]


def feat_eng_alpha_delta(
    train_pl_df: pl.DataFrame, test_pl_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """alpha * delta を新しい特徴量として追加する関数

    Args:
        train_pl_df (pl.DataFrame): train data
        test_pl_df (pl.DataFrame): test data

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: alpha * delta を追加した train, test データ
    """
    df_list = [train_pl_df, test_pl_df]
    calc_df_list = []

    for df in df_list:
        df = df.with_columns((pl.col("alpha") * pl.col("delta")).alias("alpha_delta"))

        calc_df_list.append(df)

    return calc_df_list[0], calc_df_list[1]


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
    x_train_pd = train_pl_df.drop("class").to_pandas()

    # categorical data の削除
    x_train_pd = x_train_pd.drop(CATEGORICAL_COLS, axis=1)

    # id 列の削除
    x_train_pd = x_train_pd.drop("id", axis=1)

    target_train_pd = train_pl_df.select(train_pl_df["class"]).to_pandas()
    test_pd_df = test_pl_df.to_pandas()
    # categorical data の削除
    test_pd_df = test_pd_df.drop(CATEGORICAL_COLS, axis=1)

    return (x_train_pd, target_train_pd, test_pd_df)


def create_model(verbosity: int = -1) -> lgb.LGBMClassifier:
    """LightGBMのモデルを作成する

    Returns:
        lgb.LGBMClassifier: モデルの設定
    """

    return lgb.LGBMClassifier(
        **LGB_PARAMS,
        verbosity=verbosity,
    )


def _5_fold_cv(
    x_train_pd: pd.DataFrame,
    target_train_pd: pd.DataFrame,
) -> None:
    """5-fold Cross Validation"""

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=96,
    )

    scores = []

    y = target_train_pd.values.ravel()

    for fold, (train_idx, valid_idx) in enumerate(skf.split(x_train_pd, y), start=1):
        X_train = x_train_pd.iloc[train_idx]
        X_valid = x_train_pd.iloc[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # ハイパーパラメータの設定
        model = create_model()

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(50),
            ],
        )

        pred = model.predict(X_valid)

        score = accuracy_score(y_valid, pred)

        scores.append(score)

        print(f"Fold {fold}: {score:.6f}")

    print("-" * 50)
    print(f"CV Mean : {sum(scores) / len(scores):.6f}")
    print(f"CV Std  : {pd.Series(scores).std():.6f}")
    print("-" * 50)


def train_model(x_train_pd: pd.DataFrame, target_train_pd: pd.DataFrame) -> lgb:
    """LightGBM で学習する

    Args:
        x_train_pd (pd.DataFrame): 目的変数を除外した学習データ
        target_train_pd (pd.DataFrame): 目的変数

    Returns:
        lgb: 学習モデル
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        x_train_pd,
        target_train_pd.values.ravel(),
        test_size=0.2,
        stratify=target_train_pd.values.ravel(),
        random_state=96,
    )

    # ハイパーパラメータの設定
    model = create_model(verbosity=1)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"Best iteration : {model.best_iteration_}")

    return model


def importance_feature(model: lgb.LGBMClassifier, x_train_pd: pd.DataFrame) -> None:
    """学習した model における特徴量重要度を算出する関数

    Args:
        model (gb.LGBMClassifier): LightGBM(base model)
        x_train_pd (pd.DataFrame): train data
    """
    # 特徴量重要度の取得
    importance_df = pd.DataFrame(
        {
            "feature": x_train_pd.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print(importance_df)


def predict(model: lgb.LGBMClassifier, test_pd_df: pd.DataFrame) -> pd.Series:
    """train model を用いて predict する

    Args:
        model (lgb.LGBMClassifier): train model
        test_pd_df (pd.DataFrame): test data

    Returns:
        pd.Series: class 名で出力するデータ
    """
    pred = model.predict(
        test_pd_df.drop(columns="id"),
        num_iteration=model.best_iteration_,
    )

    return pred


def create_submission_filename() -> str:
    """実行中スクリプト名を使って submission ファイル名を作成する"""

    # 実行する時のスクリプト名
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

    # 特徴量エンジニアリング(波長の差, redshift と波長の相互作用項, alpha*delta)
    train_pl_df, test_pl_df = feat_eng_spectrum(train_pl_df, test_pl_df)
    train_pl_df, test_pl_df = feat_eng_spectrum_redshift(train_pl_df, test_pl_df)
    train_pl_df, test_pl_df = feat_eng_alpha_delta(train_pl_df, test_pl_df)

    # 学習データ(classなし), class データ, test データの分割する
    x_train_pd, target_train_pd, test_pd_df = preprocess_data(train_pl_df, test_pl_df)

    # 5-fold-cv の実行
    _5_fold_cv(x_train_pd, target_train_pd)

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
