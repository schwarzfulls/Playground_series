# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "catboost",
#     "pyarrow",
#     "ruff",
#     "mypy",
# ]
# ///

"""
目的
base model(catboost) + 特徴量エンジニアリング(波長マイナス, redshift*スペクトルの差分) + カテゴリカルデータの削除

結果
予測精度Score: 0.94654

特徴量重要度
         feature  importance
7       redshift   35.149551
9            g_r   10.370620
6              z    8.796241
0          alpha    7.498992
8            u_g    5.563555
10           r_i    5.452889
1          delta    5.281749
13  redshift_g_r    4.996367
2              u    3.682106
3              g    2.979716
5              i    2.389816
11           i_z    2.195237
12  redshift_u_g    1.874908
14  redshift_r_i    1.524460
4              r    1.470777
15  redshift_i_z    0.773018

分かったこと
- 予測精度は0.00015上昇した．
  - 今回の予測では categorical data は役に立たない？
    - よりシンプルなモデルになり分岐精度が上昇している可能性がある．

次回やること
- 現在の特徴量エンジニアリング(波長の差分と redshift * 波長の差分)を作成するモデル
    - カテゴリカルデータを削除して，LightGBM の予測精度を確認する
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

    # categorical data の削除
    x_train_pd = x_train_pd.drop(CATEGORICAL_COLS, axis=1)

    # id 列の削除
    x_train_pd = x_train_pd.drop("id", axis=1)

    target_train_pd = train_pl_df.select(train_pl_df["class"]).to_pandas(
        use_pyarrow_extension_array=True
    )
    test_pd_df = test_pl_df.to_pandas(use_pyarrow_extension_array=True)
    # categorical data の削除
    test_pd_df = test_pd_df.drop(CATEGORICAL_COLS, axis=1)

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

    model.fit(x_train_pd, target_train_pd)

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

    # 特徴量エンジニアリング(波長の差, redshift の非線形化)
    train_pl_df, test_pl_df = feat_eng_spectrum(train_pl_df, test_pl_df)
    train_pl_df, test_pl_df = feat_eng_spectrum_redshift(train_pl_df, test_pl_df)

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
