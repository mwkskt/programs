from sklearn.datasets import fetch_california_housing
    # fetch_california_housing関数をインポート
from sklearn.model_selection import train_test_split
    # train_test_split 関数は、データセットを訓練セットとテストセットに分割する関数
import numpy as np
    # numpyは数値計算のライブラリ
from sklearn.tree import DecisionTreeRegressor
    # 決定木クラスをインポート
from sklearn.metrics import mean_squared_error
    # mean_squared_error関数のインポート
    # mean_squared_error
        # 二乗平均誤差を計算する
import matplotlib.pyplot as plt
    # matplotlibライブラリのpyplotモジュールをインポート
    # matplotlibはデータの視覚化（グラフ）に使うライブラリ
    # pyplot
        # グラフィックスプログラミングインターフェースを提供するモジュール
        # MATLAB風である

# データ読み込み
data = fetch_california_housing()
    # カリフォルニアの住宅価格データセットをdataとして取得
X, y = data.data, data.target
    # Xに特徴量を、yにターゲット（住宅価格）を割り当て
 
# 訓練データ、テストデータへの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    # train_test_splitはX、yを訓練データとテストデータで分割する
    #_trainがつくのはX、yそれぞれにおける訓練データ
    #_testがつくのはX、yそれぞれにおけるテストデータ
    # この場合、訓練データは全体の7割
    # random_state=0とすると、ランダムな選び方（データの分割）に再現性を確保できる

# 最良のモデルを保持する
best_mse = np.inf
    # トレーニング中に最良の検証セットの平均二乗誤差（MSE）を追跡するための変数
    # np.infはNumPyで定義される定数で、正の無限大（限界値を超える大きさ）を表す
    # 例えば最小値を探すときの初期値として便利
best_depth = 1
    # トレーニング中に最良となる決定木の深さを追跡するための変数
history = []

# 訓練のループ
for tree_depth in range(1, 50 + 1, 1):
    # tree_depthは決定木の最大深さで、1から50まで1刻みに変化する。

    # モデルの定義
    model = DecisionTreeRegressor(max_depth=tree_depth, random_state=0)
        # max_depthは決定木の最大深さを制御するパラメータ
            # ここでは、決定木の最大深さがtree_depthになるように設定
        # random_state=0は、特徴量の選び方に再現性を確保する
    
    # モデルの訓練
    model.fit(X_train, y_train)

    # モデルの評価
    y_pred = model.predict(X_test)
        # X_testをモデルに与え、予測値y_predを取得
    mse = mean_squared_error(y_test, y_pred)
        # 予測結果と正解データを使用して損失（二乗平均誤差）を計算
    mse = float(mse)
        # mseをfloatに変換
    history.append(mse)
        # リストhistoryにmseを追加し、mseの履歴を記録
    if mse < best_mse:
        # mseがbest_mseを下回っていれば以下を行う

        best_mse = mse
            # best_mseをmseに更新
        best_depth = tree_depth
            # best_depthをtree_depthに更新

print("Depth: ", best_depth)
    # best_depthを出力
print("MSE: %.2f" % best_mse)
    # best_mseを出力
    # %f
        # 引数を浮動小数点数として表示するためのプレースホルダ
        # %.2fとすると小数点以下の桁数を2桁に制限する
    # %
        # 先に出てきたプレースホルダをこの%の後に続く変数で置き換える
    # ここでは、%.2fの所を浮動小数点数であるbest_mseが置き換えた形で出力される
    # それも、小数点以下の桁数を2桁に指定される。
print("RMSE: %.2f" % np.sqrt(best_mse))
    # best_mseの正の平方根を出力

# グラフ化
plt.plot(range(1, 50 + 1, 1), history)
    # historyを平滑線でプロット
plt.xlabel('maximum depth of decision tree')
    # 横軸のラベルをepoch numberに設定
plt.ylabel('mean square error')
    # 縦軸のラベルをmean square errorに設定
plt.show()
    # Matplotlibで作成した図を表示する関数
    # グラフやチャートを画面上に描画する