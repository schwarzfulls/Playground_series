# 目的

入力データから id ごとの class を予測して，予測精度を競う.

## 提出フォーマット

id に対する class を記載して提出する

| id  | class  |
| :-: | :----: |
|  1  |  STAR  |
|  1  | GALAXY |
|  1  |  STAR  |

## カラムの意味

id(Int64): 主キー
alpha(Float64): 天球上の経度(0~360°)
delta(Float64): 天球上の緯度(-90~90°)
u(Float64): 紫外線付近の明るさ(uバンドの等級)
g(Float64): 緑色付近(g バンドの等級)
r(Float64): 赤色付近(r バンドの等級)
i(Float64): 近赤外(i バンドの等級)
z(Float64): より長波長側の近赤外(z バンドの等級)
redshift(Float64): 地球から遠ざかる速度・距離の指標
spectral_type(String): スペクトル解析から得られる分類情報
galaxy_population(String): 銀河が属する集団や分類
class(String): 予測変数, "STAR", "QSO", "GALAXY" の3種類を予測する

## 用語の整理

クエーサー(QSO)

## その他の情報

SDSS(Sloan Digital Sky Survey)の天体データの可能性がある．

u, g, r, i ,z の意味
感覚的には，同じ天体を違う色のサングラスで見た明るさ．
r, g, b ではなく，天文学ではもっと細かく色を分類して観測する．
天文学では下の順番で観測する．
天体 -> uフィルタ -> CCD
天体 -> gフィルタ -> CCD

u, g, r, i, z は等級の値を示す．
`数値が小さいほど，明るい`

色指数(組み合わせ)で考えると良さそう
波長から，class を分類するのが妥当な手段？
