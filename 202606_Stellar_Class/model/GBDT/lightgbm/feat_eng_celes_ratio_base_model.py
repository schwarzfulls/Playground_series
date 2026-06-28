# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "lightgbm",
#     "pyarrow",
#     "ruff",
#     "mypy",
# ]
# ///

"""
目的
base model(LightGBM) + 特徴量エンジニアリング(波長マイナス, redshift*スペクトルの差分) + カテゴリカルデータの削除 + alpha * delta + 波長の比率

結果Score: 0.95339

特徴量重要度
         feature  importance
0          alpha        5723
7       redshift        4579
1          delta        4519
16   alpha_delta        3001
6              z        2859
2              u        2526
3              g        2364
4              r        1841
5              i        1785
13  redshift_g_r        1728
18     g_r_ratio        1538
19     r_i_ratio        1523
8            u_g        1367
14  redshift_r_i        1362
17     u_g_ratio        1288
20     i_z_ratio        1264
12  redshift_u_g        1239
15  redshift_i_z        1232
10           r_i        1192
9            g_r        1141
11           i_z         926

分かったこと
- 今までの特徴量エンジニアリングも考慮すると非線形の特徴量の追加よりも，特徴量同士の相互関係の方がスコアに効く
  - 波長の比率は効果がない

次回やること
- alpha と delta の特徴量の組み合わせを増やしてみる
  - alpha * redshift, delta * redshift, alpha * u_g, delta * g_r
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
import polars as pl

TRAIN_DF_PATH = Path("./input/")
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]

OUTPUT_DIR = Path("./output/GBDT/lightgbm")


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
        calc_df_list.append(
            df.with_columns(
                [
                    (pl.col("redshift") * pl.col("u_g")).alias("redshift_u_g"),
                    (pl.col("redshift") * pl.col("g_r")).alias("redshift_g_r"),
                    (pl.col("redshift") * pl.col("r_i")).alias("redshift_r_i"),
                    (pl.col("redshift") * pl.col("i_z")).alias("redshift_i_z"),
                ]
            )
        )

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


def feat_eng_ratio(
    train_pl_df: pl.DataFrame, test_pl_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """波長の比率を新しい特徴量として追加する関数

    Args:
        train_pl_df (pl.DataFrame): train data
        test_pl_df (pl.DataFrame): test data

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: 波長の比率を追加した train, test データ
    """
    df_list = [train_pl_df, test_pl_df]
    calc_df_list = []
    for df in df_list:
        calc_df_list.append(
            df.with_columns(
                [
                    (pl.col("u") / pl.col("g")).alias("u_g_ratio"),
                    (pl.col("g") / pl.col("r")).alias("g_r_ratio"),
                    (pl.col("r") / pl.col("i")).alias("r_i_ratio"),
                    (pl.col("i") / pl.col("z")).alias("i_z_ratio"),
                ]
            )
        )

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


def train_model(x_train_pd: pd.DataFrame, target_train_pd: pd.DataFrame) -> lgb:
    """LightGBM で学習する

    Args:
        x_train_pd (pd.DataFrame): 目的変数を除外した学習データ
        target_train_pd (pd.DataFrame): 目的変数

    Returns:
        lgb: 学習モデル
    """
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=96,
        verbosity=1,
    )

    model.fit(
        x_train_pd,
        target_train_pd.values.ravel(),
    )

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
    pred = model.predict(test_pd_df.drop(columns="id"))

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

    # 特徴量エンジニアリング(波長の差, redshift の非線形化)
    train_pl_df, test_pl_df = feat_eng_spectrum(train_pl_df, test_pl_df)
    train_pl_df, test_pl_df = feat_eng_spectrum_redshift(train_pl_df, test_pl_df)
    train_pl_df, test_pl_df = feat_eng_alpha_delta(train_pl_df, test_pl_df)
    train_pl_df, test_pl_df = feat_eng_ratio(train_pl_df, test_pl_df)

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
