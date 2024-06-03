# NNによる回帰を行う。
    # 正規化あり
    # 最終的には二乗平均誤差の変遷をグラフ化し、ターミナル上に二乗平均誤差およびその正の平方根を出力

from sklearn.datasets import fetch_california_housing
    # fetch_california_housing関数をインポート
from sklearn.model_selection import train_test_split
    # train_test_split 関数は、データセットを訓練セットとテストセットに分割する関数
    # fromを使うのは特定のモジュールやオブジェクトを選択的にインポートできるから
from sklearn.preprocessing import StandardScaler
    # StandardScalerクラスはデータの標準化（またはZスコア正規化）に使用
import torch
    # torchはPyTorchライブラリのためのパッケージ
import torch.nn as nn
    # torch.nnはPyTorchのニューラルネットワークモジュール
import torch.optim as optim
    # torch.optimはPyTorchの最適化アルゴリズムモジュール
import numpy as np
    # numpyは数値計算のライブラリ
import tqdm
    # tqdmはプログレスバーを表示するためのライブラリ
    # 進捗状況の可視化に使う
import matplotlib.pyplot as plt
    # matplotlibライブラリのpyplotモジュールをインポート
    # matplotlibはデータの視覚化（グラフ）に使うライブラリ
    # pyplotはグラフィックスプログラミングインターフェースを提供するモジュール
        # MATLAB風である

# データ読み込み
data = fetch_california_housing()
    # カリフォルニアの住宅価格データセットをdataとして取得
X, y = data.data, data.target
    # Xに特徴量を、yにターゲット（住宅価格）を割り当て

# 訓練データ、テストデータへの分割
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7)
    # train_test_splitはX、yを訓練データとテストデータで分割する
    # _rawとしているのは、特徴量は後で標準化するため
    # この場合、訓練データは全体の7割

# データの標準化
scaler = StandardScaler()
    # StandardScalerインスタンスを作成
    # 特徴量を平均が0、分散が1になるように変換（標準化）
    # 正規化
        # データのスケール（単位）を扱いやすいものに整えること
        # 
    # 標準化
        # データの平均を0、標準偏差（データのバラツキ具合）を1に変換すること
    # Zスコア正規化は標準化でもあるので、標準化ともいう。
scaler.fit(X_train_raw)
    # fitメソッドを使って、訓練データX_train_rawの平均と標準偏差を計算
X_train = scaler.transform(X_train_raw)
    # transformメソッドを使用して、X_train_rawデータを標準化
X_test = scaler.transform(X_test_raw)
    # transformメソッドを使用して、X_test_rawデータを標準化
    # ここでの標準化はX_train_rawから計算した平均と標準偏差を使用している
    # 同じような値を持つ訓練データとテストデータが両方とも同じ値に標準化されないとモデルの性能が低く見積もられる

# Pytorchのテンソルへの変換
X_train = torch.tensor(X_train, dtype=torch.float32)
    # X_trainをPytorchのテンソルに変換し、データ型をfloat32に変換
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    # y_trainをPytorchのテンソルに変換し、データ型をfloat32に変換
    # .reshape(-1, 1）
        # 行ベクトルから列ベクトルにも変換
X_test = torch.tensor(X_test, dtype=torch.float32)
    # X_testをPytorchのテンソルに変換し、データ型をfloat32に変換
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    # y_testをPytorchのテンソルに変換し、データ型をfloat32に変換
    # .reshape(-1, 1）
        # 行ベクトルから列ベクトルにも変換

# モデルの定義
model = nn.Sequential(
    nn.Linear(8, 24),
        # 8つの入力特徴量を受け取り、それらを24個の中間層のノードに変換
        # 線形層
            # その層の全てのノードが、前の層の全てのノードと結合を持つような層のこと
            # 線形層は、重みが０だったとしても結合が存在しないのではなく影響がないだけと考え、
            # そもそも結合を定義していない状態とは区別する。
            # 全結合層ともいう。
    nn.ReLU(),
        # 活性化関数にReLUを指定
    nn.Linear(24, 12),
        # 中間層の出力（24個）を12個のノードに変換
    nn.ReLU(),
        # 活性化関数にReLUを指定
    nn.Linear(12, 6),
        # 12個の中間層のノードを6個のノードに変換
    nn.ReLU(),
        # 活性化関数にReLUを指定
    nn.Linear(6, 1)
        # 6つの中間層のノードを1つの出力ノードに変換
)

# 損失関数と最適化
loss_fn = nn.MSELoss()
    # 平均二乗誤差（MSE）を損失関数として定義
optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Adam最適化アルゴリズムを使用して、モデルのパラメータを最適化
    # 学習率は0.0001、model.parameters() は、モデル内のすべてのパラメータ（重みとバイアス）を返す
    # 学習率
        # ニューラルネットワークの訓練において、各更新ステップでパラメータ（重みやバイアス）を
        # どれだけ変更するかを決定するハイパーパラメータ

# 訓練パラメータ
n_epochs = 100
    # 実行するエポック数を100に設定
    # エポック（epoch）とは、トレーニングデータセット全体を1回処理すること
batch_size = 10
    # バッチサイズを10に設定
    # バッチサイズは、ミニバッチ勾配降下法を用いてモデルをトレーニングする際に、1度に処理されるサンプルの数
    # ミニバッチ勾配降下法では、トレーニングデータ全体を一度に処理するのではなく、バッチごとにデータを分割して処理する
batch_start = torch.arange(0, len(X_train), batch_size)
    # バッチの開始インデックスを定義
    # torch.arange() 関数を使用して、0から len(X_train) までの整数のシーケンスを作成
    # その後、バッチサイズごとにインデックスを取得
    # トレーニングデータセット全体をバッチに分割し、バッチごとにモデルをトレーニングする準備が整う
    # プログレスバーを表示するのに使う。0スタートで最大がX_trainの長さ、batch_sizeずつ変化していく

# 最良のモデルを保持する
best_mse = np.inf
    # トレーニング中に最良の検証セットの平均二乗誤差（MSE）を追跡するための変数
    # np.infはNumPyで定義される定数で、正の無限大（限界値を超える大きさ）を表す
    # 例えば最小値を探すときの初期値として便利
best_weights = None
    # 最良のモデルの重みを保存するための変数
history = []
    # mseの履歴を記録するためのリスト

# 訓練のループ
for epoch in range(n_epochs):
    # n_epochs回繰り返す

    model.train()
        # モデル（model）をトレーニングモードに設定
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        # with
            # コンテキストマネージャーを使用するためのキーワード
            # コンテキストマネージャー
                # 特定のブロックの実行前後に自動的に何らかの処理を行うことができる機能
            # この場合、withブロック（withの後インデントを使っている範囲）でのみtqdmによるプログレスバーが
            # 有効になり、withブロックを抜けるとプログレスバーが無効となる。
        # tqdmライブラリを使用して、プログレスバーを表示
            # ここではbatch_startに対してプログレスバーを作成
            # つまり、batch_startに対する進捗状況が表示される
        # unit="batch"
            # プログレスバーの単位をbatchに設定
        # mininterval=0
            # 更新間隔の最小値を0秒、すなわちできるだけ頻繁に更新するように設定
        # disable=True
            # プログレスバーの表示を無効にする
        bar.set_description(f"Epoch {epoch}")
            # epoch変数の値を使用して、現在のエポック数を表示
            # set_description()
                # プログレスバーの先頭に渡された引数を追加する
            # f"Epoch {epoch}"
                # F-string（フォーマット済み文字列リテラル）を用いてEpochを表示
                # F-stringは文頭にfをおいて定義し、{}内に変数を入れるとその位置に変数の中身を
                # 文字列として表示して置き換える。
                # 今回はEpoch {epoch}とあるので、文中の{epoch}がepochの中身である0、1、2、3・・・
                # で置き換わって表示される
            # 今回はtqdm.tqdm（as barとしているのでbarで参照できる）の引数としてdisable=Trueとしており、
            # プログレスバーの表示を無効化したためプログレスバーは表示されない。disable=Falseとすれば
            # プログレスバーが表示される。
        for start in bar:
            # barはプログレスバーであり、各イテレーションで新しいバッチの開始位置（start）を返す
            # プログレスバーはbatch_startに対して作成されており、この引数はバッチサイズごとに数字が振られているので
            # プログレスバーからイテレータによってデータを取り出せば、そのデータはバッチの開始位置を表すことになる
            # イテレータ
                # リストや辞書など、データの集合に対して順にアクセスするオブジェクト
                # 属性として現在参照しているデータ、メソッドとして次のデータを参照する機能を持つのでオブジェクト
                # Pythonでは順序を考えることのできるデータの集合を順序通りに辿っていったときにデータを全部
                # 辿り尽くすまで繰り返しを続ける方式で繰り返しを制御する。このため、繰り返しを行うためには
                # リストのようにデータの集合が必要となるが、データの集合

            # バッチの取り出し
            X_batch = X_train[start:start+batch_size]
                # startを開始位置とし、start + batch_sizeを終了位置とするスライスを使用して、
                # トレーニングデータセット（X_train）からbatch_size分の入力データ（X_batch）を取得
                    # そもそもbatch_size = 10としてデータを分割して処理することにしていた

            y_batch = y_train[start:start+batch_size]
                # startを開始位置とし、start + batch_sizeを終了位置とするスライスを使用して、
                # トレーニングデータセット（y_train）からbatch_size分の入力データ（y_batch）を取得
            
            # 順伝播
            y_pred = model(X_batch)
                # X_batchをモデルに与え、予測値y_predを取得
                # モデルは入力データを順方向に処理し、最終的な予測値を生成
            
            # 誤差（損失）の計算
            loss = loss_fn(y_pred, y_batch)
                # 実際のラベルデータであるy_batchとモデルの予測値y_predとの間で損失を計算
                # ここでは予測値とラベルデータの間の平均二乗誤差（MSE）

            # 逆伝播
            optimizer.zero_grad()
                # モデルのすべての勾配をゼロに初期化
            loss.backward()
                # 損失を使用して勾配を計算
                # 逆伝播アルゴリズムによる
    
            # 重みの更新
            optimizer.step()
                # 計算された勾配を使用してモデルの重みを更新
                
            # 進捗の表示
            bar.set_postfix(mse=float(loss))
                # プログレスバーに最新の損失値を表示
            
    # 各エポックの終わりに精度を評価
    model.eval()
        # モデルを評価モードに切り替え
    y_pred = model(X_test)
        # テストデータに対する予測を行い,予測結果をy_predに格納
    mse = loss_fn(y_pred, y_test)
        # 予測結果と正解データを使用して損失を計算
    mse = float(mse)
        # mseをfloatに変換
    history.append(mse)
        # リストhistoryにmseを追加し、mseの履歴を記録
    if mse < best_mse:
        # mseがbest_mseを下回っていれば以下を行う
        
        best_mse = mse
            # best_mseをmseに更新
        
# 数値の出力
print("MSE: %.2f" % best_mse)
    # best_mseを出力
    # %f
        # 引数を浮動小数点数として表示するためのプレースホルダ
        # %.2fとすると小数点以下の桁数を２桁に制限する
    # %
        # 先に出てきたプレースホルダをこの%の後に続く変数で置き換える
    # ここでは、%.2fの所を浮動小数点数であるbest_mseが置き換えた形で出力される
    # それも、小数点以下の桁数を2桁に指定される。
print("RMSE: %.2f" % np.sqrt(best_mse))
    # best_mseの正の平方根を出力

# グラフ化
plt.plot(history)
    # historyを平滑線でプロット
plt.xlabel('epoch number')
    # 横軸のラベルをepoch numberに設定
plt.ylabel('mean square error')
    # 縦軸のラベルをmean square errorに設定
plt.show()
    # Matplotlibで作成した図を表示する関数
    # グラフやチャートを画面上に描画する
