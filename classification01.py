# 参考サイト:https://qiita.com/Uta10969/items/a5dc0d37ebfc9ac6400b
# PytorchにてMNISTデータセットを分類

import torchvision
    # Pytorchに付属していて、画像処理に関係するライブラリ
import torchvision.transforms as transforms
    # 画像データの前処理や拡張のためのモジュール
import torch
    # torchはPyTorchライブラリのためのパッケージ
import torch.nn as nn
    # torch.nnはPyTorchのニューラルネットワークモジュール
import torch.nn.functional as F
    # ニューラルネットワークを構築する際に必要な関数群のモジュール
import torch.optim as optim
    # torch.optimはPyTorchの最適化アルゴリズムモジュール
import matplotlib.pyplot as plt
    # matplotlibはデータの視覚化（グラフ）に使うライブラリ
    # pyplotはグラフィックスプログラミングインターフェースを提供するモジュール

input_size = 28*28
    # 入力層のノード数（入力サイズ28*28ピクセル分の数値を入力データとして使う）
hidden1_size = 1024
    # 最初の隠れ層のノード数
hidden2_size = 512
    # 2番目の隠れ層のノード数
output_size = 10
    # 出力層のノード数（0〜9の10個のラベルに分類する）
num_epochs = 10
    # num_epochはエポック数として使われているのでこれを10に設定
batch_size = 250
    # ミニバッチ学習におけるミニバッチのサイズを指定
    # この場合、各イテレーションで250個のデータサンプルが与えられて学習する
    # イテレーション
        # 1つのミニバッチが訓練プロセスを通過し、重みが一度更新されるプロセス
        # 厳密には、プログラミングにおいて繰り返しの動作、あるいはその1回分の動作を指す用語なので、
        # 必ずしもNNに使う用語ではない。ここでは、学習を250回繰り返して1バッチに含まれるデータ全てを
        # 学習するという動作を表しているので、1回のイテレーションで1バッチ分の学習が行われることになる。
    # エポック
        # 訓練データセット全体が訓練プロセスを通過するプロセス
        # つまりイテレーションを訓練データ数/バッチサイズ分だけ繰り返した過程

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download = True)
    # 訓練データセットの作成
    # PytorchではMNISTデータセットはtorchvision.datasets.MNISTで提供される
    # root='./data
        # ダウンロード、もしくは読み取る際のディレクトリを指定
        # カレントディレクトリ（./）直下のdataフォルダに指定
    # train=True
        # 訓練データセットを読み取ることを指定
    # transform=transforms.ToTensor()
        # データセットに適用される変換を指定
        # PyTorchのテンソルに変換し、ピクセルの値を[0, 1]の範囲にスケーリング
    # download=True
        # Trueに設定されている場合、指定されたディレクトリにデータセットが存在しない場合にインターネットから
        # データセットをダウンロード。既にデータセットがダウンロードされている場合は、再ダウンロードしない
        
test_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download = True)
    # テストデータセットの作成
    # テストデータセットについても訓練データと同様に設定

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    # DataLoaderクラスを使用して、訓練データセット用のデータローダーを作成
    # DaraLoader
        # データセットから自動的にミニバッチを生成し、訓練中にそれらをモデルに供給する
        # ニューラルネットワークの訓練過程では下記の設定に基づいてデータが供給される
    # dataset=train_dataset
        # 使用するデータセットオブジェクト（ここではtrain_dataset）を指定
    # batch_size=batch_size
        # ミニバッチのサイズを指定
        # 先に出てくるbatch_sizeは引数のbatch_size、後に出てくるのは、
        # 先程batch_size=250として設定したbatch_size
    # shuffle=True
        # 各エポックの開始時にデータセットをランダムにシャッフルする
        # デフォルトではshuffle=Falseとなっていて、シャッフルされない
    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
    # テストデータセットについても訓練データセットと同様に設定

class Net(nn.Module):
    # nn.Moduleクラスから継承した新しいクラスNetを定義
    # nn.Module
        # PyTorchにおける全てのニューラルネットワークモジュールの基底クラス（スーパークラス）
    
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # クラスのコンストラクタ
            # Pythonではコンストラクタの名前は__int__で指定されている。

        super(Net, self).__init__()
            # Netクラスのスーパークラスであるnn.Moduleのコンストラクタを動かす
            # これによりnn.Moduleが初期化される
            # ニューラルネットワークに必要なパラメータなどが準備される
        
        self.fc1 = nn.Linear(input_size, hidden1_size)
            # 入力層と最初の隠れ層の間の結合を作成
            # input_size（特徴量の数）からhidden1_size（最初の隠れ層の数）へのニューロンの結合を持ち、
            # 全結合層を成す
            # 全結合層
                # その層の全てのノードが、前の層の全てのノードと結合を持つような層のこと
                # 全結合層は、重みが０だったとしても結合が存在しないのではなく影響がないだけと考え、
                # そもそも結合を定義していない状態とは区別する。
                # 線形層ともいう。この線形というのは線型写像によって結合しているということを
                # 意味しているわけではないので、バイアス項が存在しないこととは別の話。
            # self
                # このNetクラスのインスタンスそれ自体を表す
                # self.fc1とは、このインスタンスのfc1という属性ということ。つまりself.fc1とすることで、
                # 属性fc1をインスタンスを作ってから定義できる。例えばNetクラスのインスタンスを実際に作って
                # しまってから後出しで属性を定義したり、どのような属性を持たせるかを考えることなく逐一
                # 属性を追加するように定義していける。今回は実際にインスタンスを作ってからではないから
                # 後者の利用法。コンストラクタ内で属性を設定するのは別にC++とかでもできるが、それは
                # あらかじめどのような属性が存在するかを定義してからの話。ここでは、そもそもどんな属性が
                # 存在するかを定義せずに属性を定義している。その意味でC++とかよりも柔軟に定義できる。

        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
            # 最初の隠れ層と2番目の隠れ層の間の結合を作成
            # hidden1_size（最初の隠れ層の数）からへhidden2_size（2番目の隠れ層の数）への
            # ニューロンの結合を持ち、全結合層を成す

        self.fc3 = nn.Linear(hidden2_size, output_size)
            # 2番目の隠れ層と出力層の間の結合を作成
            # hidden2_size（2番目の隠れ層の数）からへoutput_size（出力層の数）への
            # ニューロンの結合を持ち、全結合層を成す

    def forward(self, x): 
        # self：Netクラスのインスタンス、x：入力
        # 順伝播（ニューラルネットワークでのデータの流れ）をforward関数として実装

        z1 = F.relu(self.fc1(x))
            # self.fc1(x)
                # 入力xに対して最初の全結合層fc1の計算を適用
            # F.relu(self.fc1(x))
                # fc1の計算結果にReLu関数を適用し、最初の隠れ層の値とする
        
        z2 = F.relu(self.fc2(z1))
            # self.fc2(z1)
                # 最初の隠れ層の値z1に対して2番目の全結合層fc2の計算を適用
            # F.relu(self.fc2(z1))
                # fc2の計算結果にReLu関数を適用し、2番目の隠れ層の値とする
        
        y = self.fc3(z2)
            # self.fc3(z2)
                # 2番目の隠れ層の値z2に対して3番目の全結合層fc3の計算を適用し、出力層の値とする
        
        return y
            # yを返り値として返す

model = Net(input_size, hidden1_size, hidden2_size, output_size)
    # Netクラスのインスタンスmodelを作成
    # メソッドが呼び出される際に、呼び出したインスタンス自身が自動的に渡されるためselfは引数として渡す必要はない


criterion = nn.CrossEntropyLoss()
    # 損失関数の定義
    # CrossEntropyLoss()
        # 交差エントロピー誤差関数
        # 厳密にはCrossEntropyLossというクラスが存在し、そのインスタンスcriterionを作成している
        # ()とあるのは本来与えるべき引数を与えず、デフォルト値で設定するということ
        # こうすると、与えたデータに対して、損失の平均値を返すようになる。

optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 最適化法をSGD（SGD: Stochastic Gradient Descent）に指定
    # SGD
        # 確率的勾配降下法
    # 学習率lr = 0.01に設定

def train_model(model, train_loader, criterion, optimizer):
    # train_model関数を定義
    # モデルの訓練を1エポック分だけ行い、その訓練データに対する平均的な損失を返す

    train_loss = 0.0
        #trainの損失用の変数を定義
    num_train = 0
        #学習回数の記録用の変数を定義 

    model.train()
        # モデルを学習モードに変換

    # エポック1回分の学習を行う
        # つまり、バッチサイズ分のデータで学習させることで1回パラメータを修正し
        # この過程をデータの分割数分だけ繰り返す
    for i, (images, labels) in enumerate(train_loader):
        # enumerate()
            # イテレータを引数として取り、それを反復する際に、各要素のインデックスと値をタプルとして取得する
            # イテレータ
                # 例えばリストやタプルなどの反復可能オブジェクト
            # インデックス
                # 訓練データを何番目のバッチとして処理しているかを示す数値
                # for文でのiに対応
            # 値
                # データローダーから取り出された現在のバッチにおけるデータとラベルのペア
                # for文での(images, labels)に対応
        # train_loader
            # 各バッチに対して画像データのバッチ（images）とそれに対応するラベルのバッチ（labels）を返す
            
        num_train += len(labels)
            # labelsの長さ、すなわちバッチサイズを加算していく
            # バッチサイズは任意に設定できるが、データ数をバッチサイズが割り切る保証はなく、場合によっては
            # バッチごとにバッチサイズが変わりうる。そのため、逐一バッチサイズを足し合わせていくことで
            # 訓練に用いたデータ数を正確にカウントできる。

        images, labels = images.view(-1, 28*28), labels
            # images
                # 複数の画像からなるバッチ
                # テンソルの形状は[バッチサイズ, チャネル数, 高さ, 幅]となっている
                # チャネル数
                    # 色などの情報を付与するための次元の数
                    # グレースケールだと1つ、カラー画像であればRGBの各色の分必要だから3つ
                    # 色を指定する数値じゃなくて、色を指定するために必要な数値の数（種類）
            # .view
                # PyTorchでテンソルの形状を変更する際に使用される
                # -1
                    # この手の-1は行や列を自動で調整するという命令に使われている
                    # この場合、先に-1が来ているから行を列に合わせて自動で調整する
                    # 列数はこの場合28*28
                # 元々imagesは複数の画像を一まとめにしたテンソルなので、各行が1つの画像に対応するような
                # テンソルへと変形している。これを行ごとにNNの入力とすれば画像1つ1つをNNに入力することになる。
                # そもそもMNISTの画像がグレースケールでチャネル数1、28ピクセル四方の大きさなので、
                # テンソルとしては1*28*28の要素数を持つ。これを、列数を28*28になるように行を自動調整して
                # テンソルを変形すれば、各行に画像1つ分のデータが格納されたテンソルとなる。
                
        optimizer.zero_grad()
            # 勾配を初期化

        outputs = model(images)
            # 順伝播
            # modelのメソッドとしてforwardを指定せずに引数としてimagesを渡しただけでforwardが呼び出される
            # nn.Moduleには__call()__メソッドが実装されており、これが継承先のforward()を呼び出す
            # __call()__
                # Pythonにおける特殊メソッド（マジックメソッドとも）の一つ
                # インスタンスが関数のように呼び出された時に実行される
                # ”関数のように”とは、今回であれば本来インスタンスであるはずのmodelに対して
                # model(images)のように普通の関数と同様にmodelを呼び出すということ

        loss = criterion(outputs, labels)
            # 損失の算出
            # ここでの損失は、データひとつあたりの平均値として与えられる。つまりは、outputsに含まれる画像全て
                # に対する損失を求めて、それらを画像の数で除して平均としている。つまり、最終的に得られるのは
                # データひとつあたりの損失の平均値。
        loss.backward()
            # 誤差逆伝播による勾配計算
            # .backward()メソッドはPytorchのTensorクラスに定義されている

        optimizer.step()
            # パラメータの更新
        
        train_loss += loss.item()
            # lossをtrain_losに加算代入
            # .item()
                # テンソルが1つの要素のみを含む場合に、その要素をPythonの標準的な数値型（例えば、floatやint）に
                # 変換して返す。これにより、テンソル内の数値をPythonのスカラー値として取り出せる。
    train_loss = train_loss / (num_train/batch_size)
        # num_train/batch_size
            # 訓練データ数num_trainをバッチサイズbatch_sizeで除すことでバッチ数を算出
            # MNISTデータセットの訓練データ数は60000なのでbatch_size250で割り切れる
        # train_lossは各バッチにおけるデータ一つあたりの損失の平均の合計、それをバッチ数で割れば全データに対する
        # 損失の平均になる。なぜならバッチ数で割った平均は、各バッチで得られるデータ一つあたりの損失の平均だから、
        # バッチサイズが全てのバッチで等しい限りこれが成り立つ。
        # train_lossをtrain_lossの平均値で置き換える

    return train_loss
        # train_lossを返す

def test_model(model, test_loader, criterion):
    # test_model関数
        # テストデータに対してモデルの評価を行い、平均的な損失を返す
    # test_model関数も先のtrain_model関数も基本的な動作は同じなので必要であれば参照する

    test_loss = 0.0
        # testの損失用の変数を定義
    num_test = 0
        # テスト回数の記録用の変数を定義

    model.eval()
        # modelを評価モードに変更

    with torch.no_grad():
        # 勾配計算の無効化
        # テストの時には学習に用いる勾配計算は必要ないため、無効化しておくと計算が高速化される

        for i, (images, labels) in enumerate(test_loader):
            # enumerate()
                # イテレータを引数として取り、それを反復する際に、各要素のインデックスと
                # 値をタプルとして取得する
                # イテレータ
                    # 例えばリストやタプルなどの反復可能オブジェクト
                # インデックス
                    # 訓練データを何番目のバッチとして処理しているかを示す数値
                    # for文でのiに対応
                # 値
                    # データローダーから取り出された現在のバッチにおけるデータとラベルのペア
                    # for文での(images, labels)に対応
            # test_loader
                # 各バッチに対して画像データのバッチ（images）とそれに対応するラベルのバッチ（labels）を返す
            
            num_test += len(labels)
                # labelsの長さ、すなわちバッチサイズを加算していく
                
            images, labels = images.view(-1, 28*28), labels
                # images
                    # 複数の画像からなるバッチ
                # .view
                    # PyTorchでテンソルの形状を変更する際に使用される
                    # -1
                        # この手の-1は行や列を自動で調整するという命令に使われている
                        # この場合、先に-1が来ているから行を列に合わせて自動で調整する
                        # 列数はこの場合28*28
            outputs = model(images)
                # 順伝播
                # modelのメソッドとしてforwardを指定せずに引数としてimagesを渡しただけで
                # forwardが呼び出される
            loss = criterion(outputs, labels)
                # 損失の算出
                # ここでの損失は、データひとつあたりの平均値として与えられる。つまりは、outputsに含まれる画像全て
                # に対する損失を求めて、それらを画像の数で除して平均としている。つまり、最終的に得られるのは
                # データひとつあたりの損失の平均値。
            test_loss += loss.item()
                # lossをtrain_losに加算代入
                # .item()
                    # テンソルが1つの要素のみを含む場合に、その要素をPythonの標準的な数値型（例えば、
                    # floatやint）に変換して返す。これにより、テンソル内の数値をPythonのスカラー値
                    # として取り出せる。
        test_loss = test_loss / (num_test/batch_size)
        # num_test/batch_size
            # 訓練データ数num_testをバッチサイズbatch_sizeで除すことでバッチ数を算出
            # MNISTデータセットのテストデータ数は10000なのでbatch_size250で割り切れる
        # test_lossは各バッチにおけるデータ一つあたりの損失の平均の合計、それをバッチ数で割れば全データに対する
        # 損失の平均になる。なぜならバッチ数で割った平均は、各バッチで得られるデータ一つあたりの損失の平均だから、
        # バッチサイズが全てのバッチで等しい限りこれが成り立つ。
            # test_lossをlossの平均値で置き換える
    return test_loss
        # test_lossを返す

def learning(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    # learning関数を定義

    train_loss_list = []
        # 訓練データに対する損失を記録するリストを定義
    test_loss_list = []
        # テストデータに対する損失を記録するリストを定義

    for epoch in range(1, num_epochs+1, 1):
        # epoch数分繰り返す

        train_loss = train_model(model, train_loader, criterion, optimizer)
            # train_model関数を用いて訓練データに対する損失を算出
            # train_model関数はモデルの訓練を1エポック分だけ行い、その訓練データに対する平均的な損失を返す
        test_loss = test_model(model, test_loader, criterion)
            # test_model関数を用いて訓練データに対する損失を算出
            # test_model関数はテストデータに対してモデルの評価を行い、平均的な損失を返す
        # ここでの損失は、各バッチごとに算出された損失をそのバッチのサイズで除した平均をそのエポックでの
        # 損失として評価しているが、今までは各エポックごとの学習に対する損失をそのまま損失としていた点が異なる。

        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}" 
              .format(epoch, train_loss, test_loss))
            # epoch、train_loss、test_lossを表示する。
            # {}
                # プレースホルダ
                    # 文字列の中で値を動的に挿入するために予め確保された位置やスペース
                # formatメソッドに渡された引数で置き換わる
                # .5fとしておくと、小数点以下５桁までの精度で表示される
                # .format以前は文字列のテンプレート
        train_loss_list.append(train_loss)
            # train_lossをtrain_loss_listに追加
        test_loss_list.append(test_loss)
            # test_lossをtest_loss_listに追加

    return train_loss_list, test_loss_list
        # train_loss_listとtest_loss_listを返す

train_loss_list, test_loss_list = learning(model, train_loader, test_loader, 
                                           criterion, optimizer, num_epochs)
    # train_loss_listとtest_loss_listをlearning関数の返り値として作成

plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
    # range(len(train_loss_list)), train_loss_list
        # 横軸をtrain_los_listの長さ、すなわちエポック数に、縦軸をtrain_loss_listにした
        # グラフ（平滑線）を作成
    # c='b'
        # 線の色を青に設定
    # label='train loss'
        # ラベルとしてtrain lossを設定
plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
    # range(len(test_loss_list)), test_loss_list
        # 横軸をtest_los_listの長さ、すなわちエポック数に、縦軸をtest_loss_listにした
        # グラフ（平滑線）を作成
    # c='r'
        # 線の色を赤に設定
    # label='test loss'
        # ラベルとしてtest lossを設定
plt.xlabel("epoch")
    # 横軸のラベルをepochに設定
    # predictionと統一するためにepoch numberの方がいいかも
plt.ylabel("loss")
    # 縦軸をlossに設定
plt.legend()
    # グラフに凡例（ラベル）を表示
plt.grid()
    # グラフにグリッド線を表示
plt.show()
    # 作成したグラフを表示

plt.figure(figsize=(20, 10))
    # plt.figureで新しい図を作成
    # figsizeで幅20インチ、高さ10インチの図を作成
for i in range(10):
    # 10回繰り返すことで画像10枚に対して処理を行う
    image, label = test_dataset[i]
        # test_datasetのi番目の画像とラベルをそれぞれimageとlabelに取得する
    image = image.view(-1, 28*28)
        # 画像のサイズを28*28列のテンソルへと変換
    prediction_label = torch.argmax(model(image))
        # model(image)
            # モデルに画像を入れて予測を行う
        # torch.argmax
            # 最も確率が高いラベルを取得
    # ax = plt.subplot(1, 10, i+1)
    ax = plt.subplot(2, 5, i+1)  # 2行5列のサブプロット
        # 1行10列のサブプロットを作成し、現在の画像をi+1番目の位置に配置
    plt.imshow(image.detach().numpy().reshape(28, 28), cmap='gray')
        # plt.imshowを使って画像を表示
        # plt.imshow()
            # 画像データをグラフとして表示する
            # 数値データをヒートマップの画像として表示したり、機械学習のために加工された画像データを
            # そのまま画像として表示できる。なおここでのヒートマップはデータ分析において行列型の数字データを
            # 表示させる手法のこと。
        # detach().numpy().reshape(28, 28)
            # detach()
                # 勾配情報を分離
            # .numpy()
                # NumPy配列に変換
            # reshape(28, 28)
                # 28*28の2次元配列に変形
            # 画像テンソルをNumPy配列に変換し、元の28*28ピクセルの画像に戻す
        # cmap='gray'としてグレースケールで表示
    ax.axis('off')
        # 軸（目盛やラベル）を非表示
    ax.set_title('label : {}\n Prediction : {}'.format(label, prediction_label), fontsize=15)
        # 画像の上にラベル及び予測ラベルをタイトルとして表示
        # {}
            # プレースホルダー
                # 文字列の中で後から具体的な値を挿入するために予め確保された位置を示す
        # \n
            # エスケープシーケンス
                # プログラミング言語において特殊な文字列操作や書式設定を行うための一連の文字
            # \n は改行（新しい行への移動）を意味するエスケープシーケンス
        # .format(label, prediction_label)
            # プレースホルダにlabelとprediction_labelをこの順で当てはめて表示する
        # fontsize=15にて、フォントサイズを15に指定

plt.show()
    # 全てのサブプロットを含む図を表示