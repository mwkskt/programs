# 潜在空間2次元、プロットも2次元
import torch
    # torchはPyTorchライブラリのためのパッケージ
import torch.nn as nn
    # torch.nnはPyTorchのニューラルネットワークモジュール
from torchvision import datasets, transforms
    # torchvision.datasets
        # 多くの公開データセットに簡単にアクセスできるようにするモジュール
    # torchvision.transforms
        # 画像データに対する前処理やデータ拡張を行うためのメソッド群を提供するモジュール
from torch.utils.data import DataLoader
    # DataLoader
        # データセットからデータを効率的に読み込むためのイテレータを提供するクラス
import torch.optim as optim
    # torch.optimはPyTorchの最適化アルゴリズムモジュール
import matplotlib.pyplot as plt
    # matplotlibはデータの視覚化（グラフ）に使うライブラリ
    # pyplotはグラフィックスプログラミングインターフェースを提供するモジュール
import numpy as np
    # numpyは数値計算のライブラリ

# オートエンコーダの定義
class Autoencoder(nn.Module):
    # nn.Moduleクラスを継承してAutoencoderクラスを定義

    def __init__(self):
        # コンストラクタを定義

        super(Autoencoder, self).__init__()
            # スーパークラスのコンストラクタを起動してnn.Moduleを初期化
            # super()
                # スーパークラスを参照する
            # super(Autoencoder, self)
                # Python2との互換性を意識した書き方で、Python3ではsuper().__init__()で良い
                # Python2では多重継承（複数のクラスを継承すること）した際にスーパークラスかを判別する機能が
                # 不十分であり、スーパークラスを決定できるがどのクラスのスーパークラスかを判別できなかった。
                # より正確には、引数として与えられたクラスをもとに、それが継承しているクラスの中からメソッド解決順序
                # という手順に従ってどのスーパークラスかを決めていた。
                # Python3では引数なしでも関連しているスーパークラスを参照できるようになり、引数が必要無くなった。

        # エンコーダの定義
        self.encoder = nn.Sequential(
            # nn.Sequential
                # Pytorchで提供されるクラス
                # 複数の層や伝達関数からなるNNを一つの関数と同様に扱えるようにする
                # nn.Sequentialクラスのインスタンスとしてself.encoderを作成し、
                # そのコンストラクタにNNの構成要素を引数として与えている
            nn.Linear(28*28, 128),
                # 入力層：MNISTのデータサイズ28*28に合わせてノードを作成
                # 最初の隠れ層：128個のノード
            nn.ReLU(),
                # 伝達関数をReLUに指定
            nn.Linear(128, 64),
                # ノード数128個の隠れ層をノード数64個の隠れ層に繋ぐ
            nn.ReLU(),
                # 伝達関数をReLUに指定
            nn.Linear(64, 2)
                # ノード数64個の隠れ層から出力層に繋ぐ
                # 最終的な出力層のノードが2つ、つまり潜在空間が2次元になっている
        )

        # デコーダの定義
        self.decoder = nn.Sequential(
            # nn.Sequential
                # Pytorchで提供されるクラス
                # 複数の層や伝達関数からなるNNを一つの関数と同様に扱えるようにする
                # nn.Sequentialクラスのインスタンスとしてself.decoderを作成し、
                # そのコンストラクタにNNの構成要素を引数として与えている
            nn.Linear(2, 64),
                # エンコードされた値2つを入力としてノード64個の隠れ層に繋ぐ
            nn.ReLU(),
                # 伝達関数をReLuに指定
            nn.Linear(64, 128),
                # ノード数64個の隠れ層をノード数128個の隠れ層に繋ぐ
            nn.ReLU(),
                # 伝達関数をReLuに指定
            nn.Linear(128, 28*28),
                # ノード数128個の隠れ層からノード数28*28個の出力層に繋ぐ
                # これで元の画像サイズと同じ28*28のサイズに戻る
            nn.Sigmoid()  
                # 伝達関数にシグモイド関数を指定
                # ここだけシグモイド関数なのは出力を[0, 1]の範囲に制限するため
                # ReLU関数は出力が0以上を満たすが、1以上になりうるため、[0, 1]で出力されるシグモイド関数が好適
                # 入力画像も正規化して[0, 1]の範囲に収まっている
        )

    def forward(self, x):
        # 入力データxに対してエンコードとデコードを実行し、それぞれの結果を返す
        # forwardメソッドはモデルの実行時に自動的に呼び出される
        encoded = self.encoder(x)
            # 入力データをエンコーダに通し、潜在空間表現を得る
        decoded = self.decoder(encoded)
            # 潜在空間表現をデコーダに通し、出力を得る
        return decoded, encoded
            # 出力、潜在空間表現を返す

# データセットの準備
transform = transforms.Compose([
    # transforms.Compose
        # 複数の変換をまとめて適用できるようにするクラス
        # リスト形式で渡された変換を、入力データに対して順番に実行する

    transforms.ToTensor()
        # NumPy配列をPyTorchのテンソルに変換した上で0から1の範囲に正規化
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # PytorchではMNISTデータセットはtorchvision.datasets.MNISTで提供される
    # root='./data
        # ダウンロード、もしくは読み取る際のディレクトリを指定
        # カレントディレクトリ（./）直下のdataフォルダに指定
    # train=True
        # 訓練データセットを読み取ることを指定
    # download=True
        # Trueに設定されている場合、指定されたディレクトリにデータセットが存在しない場合にインターネットから
        # データセットをダウンロード。既にデータセットがダウンロードされている場合は、再ダウンロードしない
    # transform=transform
        # データセットに適用される変換を指定
        # このtransformは、transforms.Compose()で指定した変換、すなわちPyTorchのテンソルに変換し、
        # ピクセルの値を[0, 1]の範囲にスケーリングすることを表す

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # DataLoader
        # PyTorchでデータのローディングとバッチ処理を管理するクラス
    # train_dataset
        # ロードするデータセットにtrain_datasetを指定
    # batch_size=64
        # バッチサイズを64に指定
    # shuffle=True
        # データのシャッフルを有効にする

# モデル、損失関数、オプティマイザの設定
model = Autoencoder()
    # Autoencoderクラスのインスタンスを作成
criterion = nn.MSELoss()
    # 損失関数として平均二乗誤差を指定
optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Adamオプティマイザを使用してモデルのパラメータを更新
    # Adam（Adaptive Moment Estimationの意）
        # 適応的モーメント推定のことを表し、以下の特徴を持つ
            # 各パラメータの勾配の一次モーメント（平均）と二次モーメント（分散）を計算し、
            # これらの情報を使って個々の学習率を調整する。
            # 学習初期におけるモーメントの推定が偏りがちな問題を修正する（バイアス修正）。
    # model.parameters()
        # 更新を行うパラメータ（重みやバイアス）の集合を返す
    # lr=1e-3
        # 学習率を0.001に設定
        # より正確には、基本学習率といって、各パラメータに対する具体的な学習率を決定するための初期値を設定している

# モデルの学習
def train_autoencoder(model, data_loader, criterion, optimizer, epochs=10):
    # オートエンコーダーの訓練を行うtrain_autoencoder関数を定義
    model.train()
        # modelを訓練モードに設定
        # 訓練モードでは、ドロップアウトやバッチ正規化が機能する
        # ドロップアウト
            # ニューラルネットワークの訓練中にネットワークのユニット（ノードのまとまりのこと）をランダムに
            # 無効化することで、モデルが特定の特徴に過剰に依存するのを防ぎ、モデルの過学習や局所的な最小値
            # に陥るのを防ぐ手法。
            # ドロップアウト層と呼ばれる層を設定しておき、そのうち幾らかのノードの出力を無効化する。
            # この無効化されるノードの割合をドロップアウト率と呼び、ハイパーパラメータである。
            # なお、ドロップアウトは自動的に機能するわけではなく、設定した時に使われる。
        # バッチ正規化
            # 各層の入力を正規化することで、訓練プロセスを安定させ、学習速度を向上させる手法
            # このほか、正規化されないため特定の出力の影響を受けにくくなり過学習の軽減にも有効
            # 通常、全結合層と伝達関数の間にこのバッチ正規化を行うバッチ正規化層を入れて実装する
    for epoch in range(epochs):
        # 指定されたエポック数だけループを回す
        total_loss = 0
            # 損失の合計値を0で初期化
        for data, _ in data_loader:
            # data_loaderからバッチ単位でデータを取り出してdataとする
            # dataが取り出せる間繰り返す
            # _
                # Pythonの慣例として、特定の値を無視する際に使用されるプレースホルダー
                # プレースホルダー
                    # 値が後で指定されることを期待している一時的な保持場所や、使用される予定のない値を捨てるための変数
                # この場合、訓練データに含まれるラベルを無視することを表す
            data = data.view(data.size(0), -1)
                # view()
                    # PyTorchでテンソルの形状を変更するメソッド
                # data.size(0)
                    # 現在のバッチに含まれるデータ点（サンプル）の数を取得する
                # -1
                    # 残りの次元を自動的に計算するよう指示する
                    # この場合、行数をバッチのデータ数にしているから各行は画像１つ分のデータを含む
            optimizer.zero_grad()
                # 勾配を初期化
            reconstructed, _ = model(data)
                # modelにdataを渡すと、model.forwardが自動的に呼び出されてそれにdataが渡される
                # この時、エンコーダーとデコーダーの両方から返り値が返ってくるので、
                # 必要ない方を_で受け取って無視する
                # model.forwardは、デコーダーの出力decoded、エンコーダーの出力encodedの順で返すので、
                # この場合エンコーダーの出力が無視され、デコーダーによって再構成されたデータが得られる

            loss = criterion(reconstructed, data)
                # デコーダーで再構成されたデータと元のデータとの間の損失を計算
            loss.backward()
                # 誤差逆伝播による勾配計算
                # .backward()メソッドはPytorchのTensorクラスに定義されている
            optimizer.step()
                # パラメータの更新
            total_loss += loss.item()
                # loss.item()
                    # loss（損失）は通常、PyTorchのTensorオブジェクトとして得られるので、そこから
                    # Pythonの標準的な数値（この場合はfloat）を取り出す
                # この損失を１バッチごとにtotal_lossに加算代入し、これがエポック全体で繰り返されるので
                # 最終的にtotal_lossにはエポック全体に対する損失の合計値が得られる
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}')
            # エポックと損失を表示
            # epochは0からスタートするので、そのズレを補正するために１を加える
            # 損失（Loss）は、データローダーの長さ、つまり１エポックを分割しているバッチの数
            # でエポック全体に対する損失の合計であるtotal_lossを除しているので、
            # １バッチあたりの平均的な損失を表示している。

train_autoencoder(model, train_loader, criterion, optimizer, epochs=10)
    # train_autoencoder関数を実行

# 潜在空間のプロット
def plot_latent_space(model, data_loader):
    # 潜在空間をプロットするplot_latent_space関数を定義
    model.eval()
        # モデルを評価モードに設定
        # ドロップアウトは無効化され、バッチ正規化層は学習中に計算されて固定された統計値（平均と分散）を使用
        # する。これにより、訓練中に確立されたモデルの状態でデータを一貫して評価できるようになる
    plt.figure(figsize=(10, 8))
        # 幅10インチ、高さ8インチの図を作成
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
        # plt.cm.viridis
            # matplotlibのカラーマップの一つ
        # np.linspace(0, 1, 10)
            # 指定された開始点（0）から終了点（1）までの範囲を均等に分割した値の配列を生成する
            # これらの値をカラーマップviridisに適用することで、viridisから10種類の異なる色を取り出して
            # 配列colorsとして保存できる
    with torch.no_grad():
        # with
            # コンテキストマネージャーを使用するためのキーワード
            # コンテキストマネージャー
                # 特定のブロックの実行前後に自動的に何らかの処理を行うことができる機能
            # この場合、withブロック（withの後インデントを使っている範囲）でのみtorch.no_grad()が
            # 有効になり、withブロックを抜けるとtorch.no_grad()が無効となる
        # torch.no_grad()
            # PyTorchのコンテキストマネージャーの一つ
            # モデルの評価においては不要な勾配計算を無効化して計算を高速化する
            # この勾配は本来誤差逆伝播法に用いられる
        for data, labels in data_loader:
            # data_loaderからバッチ単位でデータとラベルを取り出してdataおよびlabelsとする
            # dataとlabelsが取り出せる間繰り返す
            data = data.view(data.size(0), -1)
                # view()
                    # PyTorchでテンソルの形状を変更するメソッド
                # data.size(0)
                    # 現在のバッチに含まれるデータ点（サンプル）の数を取得する
                # -1
                    # 残りの次元を自動的に計算するよう指示する
                    # この場合、行数をバッチのデータ数にしているから各行は画像１つ分のデータを含む
            _, latent = model(data)
                # _
                    # Pythonの慣例として、特定の値を無視する際に使用されるプレースホルダー
                # modelにdataを渡すと、model.forwardが自動的に呼び出されてそれにdataが渡される
                # この時、エンコーダーとデコーダーの両方から返り値が返ってくるので、
                # 必要ない方を_で受け取って無視する
                # model.forwardは、デコーダーの出力decoded、エンコーダーの出力encodedの順で返すので、
                # この場合デコーダーによって再構成されたデータが無視され、エンコーダーの出力が得られる
            for i in range(10):
                # iを0から9までの10回繰り返す
                idx = labels == i
                    # labelsの要素のうちiに等しいもの、すなわちlabels == iが成立する要素のみを
                    # Trueとするブールインデックスを作成する。
                    # ブールインデックス
                        # ある条件に基づいて配列の要素を選択するために使用されるブール値（TrueまたはFalse）の配列
                        # 条件に一致する要素だけを抽出するのに使われる
                    # このブールインデックスによってiと等しい要素だけを抽出できるので、それを0から9まで繰り返せば
                    # 配列の要素を数字ごとにプロットできるようになる
                plt.scatter(latent[idx, 0], latent[idx, 1], color=colors[i], label=str(i))
                    # idx
                        # 0から9の特定の数字のデータ（より正確にはそれを格納している引数）に対してのみTrue
                    # latant[idx, ]
                        # latentは潜在空間表現を格納しており、idx = Trueの要素のみ抽出される
                        # latent[idx, 0]
                            # idx = Trueである要素の第一成分であり、プロットのx座標に指定
                        # latent[idx, 1]
                            # idx = Trueである要素の第二成分であり、プロットのy座標に指定
                    # color=colors[i]
                        # 点の色を配列colorsのi番目の色に指定
                        # colors[i]はカラーマップviridisから取り出した色10種を保存している
                    # label=str(i)
                        # iに保存された数値を文字列に変換し、ラベルに指定
            break
                # 最初のバッチ分が終わったら繰り返しを抜けて終わる
                
    plt.xlabel('Latent Variable 1')
        # 横軸のラベルをLatent Variable 1に指定
    plt.ylabel('Latent Variable 2')
        # 縦軸のラベルをLatent Variable 2に指定
    plt.title('2D Latent Space')
        # タイトルを2D Latent Spaceに指定
    plt.legend(title='Digits', loc='upper right')
        # 凡例を追加
        # 凡例のタイトルはDigits（数字の意）、場所は右上
    plt.show()
        # 画像を表示

test_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    # test_loaderとしてデータローダーを作成
    # なお、適用しているデータセットはtrain_datasetである点に注意
    # batch_size=1000
        # バッチサイズを1000に指定
    # shuffle=True
        # エポックの開始時にデータセットをシャッフル
        # データローダーはモデルの訓練時など、モデルがデータを必要とするタイミングで
        # データをバッチサイズ分だけ取り出してモデルに渡す
plot_latent_space(model, test_loader)
    # plot_latent_space関数を実行