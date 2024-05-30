from sklearn.datasets import fetch_openml
    # fetch_openml関数は、OpenMLデータベースからオンラインで公開されているデータセットを取得する関数
from sklearn.model_selection import train_test_split
    # train_test_split 関数は、データセットを訓練セットとテストセットに分割する関数
import numpy as np
    # numpyは数値計算のライブラリ
from sklearn.tree import DecisionTreeClassifier
    # 決定木クラスをインポート
from sklearn.metrics import log_loss
    # log_loss関数は交差エントロピー損失を計算する関数
import pickle
    # Pythonオブジェクトを直列化（シリアライズ）および逆シリアライズ（デシリアライズ）するためのツール
    # 直列化
        # Pythonオブジェクトをバイト列に変換するプロセス
    # 逆シリアライズ
        # バイト列を元のPythonオブジェクトに戻すプロセス
import matplotlib.pyplot as plt
    # matplotlibライブラリのpyplotモジュールをインポート
    # matplotlibはデータの視覚化（グラフ）に使うライブラリ
    # pyplot
        # グラフィックスプログラミングインターフェースを提供するモジュール
        # MATLAB風である

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # MNISTデータセットの読み込み
    # fetch_openml関数
        # オンラインで公開されているOpenMLデータセットを検索し、ダウンロードする
    # mnist_784
        # MNISTデータセットのこと
        # 784は、28*28 = 784ピクセルの画像を784次元のベクトルとして平坦化した形式ということ
    # version=1
        # 取得するデータセットのバージョンを指定
        # データセットは時間とともに更新されることがあるため特定のバージョンを選択できる
    # return_X_y=True
        # return_X_y=Trueとすると、fetch_openml関数はデータセットを特徴量とラベルに
        # 分けて2つのオブジェクトとして返す
        # Xはデータセットの特徴量（この場合は手書き数字の画像データ）
        # yはそれに対応するラベル（この場合は数字のクラス）を格納

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, 
                                                    random_state=0)
    # train_test_splitはX、yを訓練データとテストデータで分割する
    # MNISTデータセットは通常訓練データ60000枚、テストデータ10000枚に分割されているが、fetch_openml関数で
    # 入手するとこのように分割されない。
    #_trainがつくのはX、yそれぞれにおける訓練データ
    #_testがつくのはX、yそれぞれにおけるテストデータ
    # この場合、訓練データは全体の7割
    # random_state=0とすると、ランダムな選び方（データの分割）に再現性を確保できる

# 最良のモデルを保持する
best_loss = np.inf
    # トレーニング中に最良の検証セットの損失（loss）を追跡するための変数
    # np.infはNumPyで定義される定数で、正の無限大（限界値を超える大きさ）を表す
    # 例えば最小値を探すときの初期値として便利
best_depth = 1
    # トレーニング中に最良となる決定木の深さを追跡するための変数
train_loss_history = []
    # 訓練データに対する損失を記録するリスト
test_loss_history = []
    # テストデータに対する損失を記録するリスト

# 訓練のループ
for tree_depth in range(1, 20+1, 1):
    # tree_depthは決定木の最大深さで、1から20まで1刻みに変化する。

    # モデルの定義
    model = DecisionTreeClassifier(max_depth=tree_depth, random_state=0)
        # max_depthは決定木の最大深さを制御するパラメータ
            # ここでは、決定木の最大深さがtree_depthになるように設定
        # random_state=0は、特徴量の選び方に再現性を確保する

    # モデルの訓練
    model.fit(X_train, y_train)

    predictions_proba_train = model.predict_proba(X_train)
        # predict_proba()
            # クラスごとの予測確率を返す
            # 具体的には、決定木のリーフノードに分類されたデータがどの確率でどの
            # クラスに分類されるのかを示す。
            # この場合はX_train
        # 交差エントロピー誤差は、分類のクラスに予測した確率を、対数をとって負号を
        # つけて平均するので、あらかじめ確率を求めておく
    predictions_proba_test = model.predict_proba(X_test)
        # X_testについても同様に予測確率を計算する

    # 損失の計算
    loss_train = log_loss(y_train, predictions_proba_train)
        # 訓練データに対する交差エントロピー誤差を計算
    loss_test = log_loss(y_test, predictions_proba_test)
        # テストデータに対する交差エントロピー誤差を計算

    loss_train = float(loss_train)
        # loss_trainをfloat型に変換
    loss_test = float(loss_test)
        # loss_testをfloat型に変換

    train_loss_history.append(loss_train)
        # リストtrain_loss_historyにloss_trainを追加し、履歴として記録
    test_loss_history.append(loss_test)
        # リストtest_loss_historyにloss_testを追加し、履歴として記録

    if loss_test < best_loss:
        # loss_testがbest_lossを下回っていれば以下を行う
        # 最良の判断基準はテストデータに対する損失の小ささ
        best_loss = loss_test
            # best_lossをloss_testに更新
        best_depth = tree_depth
            # best_depthをtree_depthに更新
        best_model = model
            # best_modelをmodelで更新し、modelを保存する。
        model_pickle = pickle.dumps(best_model)
            # モデルを直列化して保存

print("Depth: ", best_depth)
    # best_depthを出力
print("Loss: %.5f" % best_loss)
    # best_lossを出力
    # %f
        # 引数を浮動小数点数として表示するためのプレースホルダ
        # %.5fとすると小数点以下の桁数を5桁に制限する
    # %
        # 先に出てきたプレースホルダをこの%の後に続く変数で置き換える
    # ここでは、%.2fの所を浮動小数点数であるbest_mseが置き換えた形で出力される
    # それも、小数点以下の桁数を2桁に指定される。

# グラフ化
plt.plot(range(1, len(train_loss_history)+1, 1), train_loss_history, c='b', label='train loss')
# plt.plot(range(1, 20+1, 1), train_loss_history, c='b', label='train loss')
    # range(1, len(train_loss_history)+1, 1)
        # 横軸をtrain_los_listの長さ、すなわち木の深さに、縦軸をtrain_loss_listにした
        # グラフ（平滑線）を作成
    # c='b'
        # 線の色を青に設定
    # label='train loss'
        # ラベルとしてtrain lossを設定
plt.plot(range(1, len(test_loss_history)+1, 1), test_loss_history, c='r', label='test loss')
    # clasification01.pyなどと同様の記法を使っているが、range関数の都合上1から20まで作ろうとすると
    # 始点と終点の指定が必要になる
# plt.plot(range(1, 20+1, 1), test_loss_history, c='r', label='test loss')
    # range(len(test_loss_list)), test_loss_list
        # 横軸をtest_los_listの長さ、すなわちエポック数に、縦軸をtest_loss_listにした
        # グラフ（平滑線）を作成
    # c='r'
        # 線の色を赤に設定
    # label='test loss'
        # ラベルとしてtest lossを設定
plt.xlabel('maximum depth of decision tree')
    # 横軸のラベルをmaximum depth of decision treeに設定
plt.ylabel('loss')
    # 縦軸のラベルをlossに設定
plt.xlim(0, 20)
    # 横軸の範囲を0から20に設定
plt.xticks(np.arange(0, 21, 5))
    # 横軸の目盛りを0から20まで5間隔に設定
    # np.arange(0, 21, 5)
        # 0から20まで5刻みの数値によるNumpy配列を作成
        # 厳密には0で始まり、21を超えない数値を5刻みで含む数値で作られている
        # plt.xticksはNumpy配列により目盛りの位置を指定できる
plt.grid()
    # グラフにグリッド線を表示
plt.legend()
    # グラフに凡例（ラベル）を表示
plt.show()
    # Matplotlibで作成した図を表示する関数
    # グラフやチャートを画面上に描画する

model = pickle.loads(model_pickle)
    # model_pickleとして保存したモデルを読み込む

plt.figure(figsize=(20, 10))
    # plt.figureで新しい図を作成
    # figsizeで幅20インチ、高さ10インチの図を作成
for i in range(10):
    # 10回繰り返すことで画像10枚に対して処理を行う
    prediction_label = model.predict(X_test.iloc[[i]])
        # predict()
            # 入力された特徴量に基づいてクラスラベルを予測
            # この場合の特徴量は画像（より正確には画像を構成するピクセル値）
        # iloc[[i]]
            # データフレームのi番目の行を抽出し、それをデータフレームとして返す
            # [i]とするとデータフレームのi番目の行をシリーズとして返す
            # 多くの機械学習アルゴリズムが入力として2次元配列を必要とするためデータフレームを入力とすることが多い
            # データフレーム
                # 2次元のラベル付きデータ構造
                # 行と列があり、それぞれにラベル（行ラベルはインデックス、列ラベルはカラム名）が付けられる
            # シリーズ
                # データフレームの一つの列に相当する1次元のラベル付き配列
                # ここでのラベルとは、インデックスと同義で、0から始まる整数値に限らず任意に設定できる
                # インデックスがあり、このインデックスによってデータの各要素にアクセスする
    ax = plt.subplot(2, 5, i+1)  # 2行5列のサブプロット
    # ax = plt.subplot(1, 10, i+1)
        # 1行10列のサブプロットを作成し、現在の画像をi+1番目の位置に配置
    plt.imshow(X_test.iloc[[i]].values.reshape(28, 28), cmap='gray')
        # plt.imshowを使って画像を表示
        # plt.imshow()
            # 画像データをグラフとして表示する
            # 数値データをヒートマップの画像として表示したり、機械学習のために加工された画像データを
            # そのまま画像として表示できる。なおここでのヒートマップはデータ分析において行列型の数字データを
            # 表示させる手法のこと。
        # iloc[[i]]
            # データフレームのi番目の行を抽出し、それをデータフレームとして返す
            # この場合X_testに入っているのはMNISTデータセットをもとに正規化した数値なのでカラムの情報は必要ない。
            # よって、iloc[i]としてシリーズ（単なる数値が格納された配列）として渡しても問題なく動作する。
        # values
            # データフレームをNumPy配列に変換
            # plt.imshow()を使うにはNumPy配列でないとといけないため、ここでNumPy配列に直す
        # reshape(28, 28)
            # NumPy配列を28*28の2次元配列に再形成
            # 元の画像データが28*28だったからそれに直す
        # cmap='gray'
            # グレースケールで表示するように設定
    ax.axis('off')
        # 軸（目盛やラベル）を非表示
    ax.set_title('label : {}\n Prediction : {}'.format(y_test.iloc[i], prediction_label[0]), fontsize=15)
        # 画像の上にラベル、予測ラベルをタイトルとして表示
        # {}
            # プレースホルダー
                # 文字列の中で後から具体的な値を挿入するために予め確保された位置を示す
        # \n
            # エスケープシーケンス
                # プログラミング言語において特殊な文字列操作や書式設定を行うための一連の文字
            # \n は改行（新しい行への移動）を意味するエスケープシーケンス
        # .format(y_test.iloc[i], prediction_label[0])
            # プレースホルダにy_test.iloc[i], prediction_label[0]をこの順で当てはめて表示する
        # prediction_label[0]
            # prediction_labelの先頭の要素、すなわち予測されたラベルを出力
            # 元々'5'などのようにシングルクォーテーションが表示されたためにint(prediction_label)として
            # いたはずなのだが、無くしてもちゃんと動いてくれたので削除した        
        # fontsize=15にて、フォントサイズを15に指定
plt.show()
    # 全てのサブプロットを含む図を表示