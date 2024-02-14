# これはニューラルネットワークに入門するためのスタート
# ここからは下記を参考サイトとしていく。
# https://atmarkit.itmedia.co.jp/ait/series/18508/

# ここではまたiris（アヤメ）を利用
# 次の手順で進める。
#
# 1.データセットの準備と整形
# 2.ニューラルネットワークの定義
# 3.学習（訓練）と制度の検証

# ------------------------------------------------------------------------------
# 1.データセットの準備と整形
# あやめのデータセットの読み込み
# ------------------------------------------------------------------------------
from sklearn.datasets import load_iris
iris = load_iris()
#print(iris)
data = iris.data
target = iris.target

#print("data = shape = \n", data.shape)
#print(data)
#print("\n")
#print("target = shape = \n", target.shape)
#print(target)
#print("feature_names = ", iris.feature_names)
#print("target_names = ", iris.target_names)

# これ、参考にできるコードだね。zip でまとめて、一緒にループできるんだ。
"""
for idx, item in enumerate(zip(iris.data, iris.target)):
    if idx == 5:
        break
    print('data:', item[0], ', target:', item[1])
"""

from sklearn.model_selection import train_test_split
"""
print('length of iris.data:', len(iris.data))  # iris.dataのデータ数
print('length of iris.target:', len(iris.target))  # iris.targetのデータ数
"""

# iris.dataとiris.targetに含まれるデータをシャッフルして分割
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
"""
print('length of X_train:', len(X_train))
print('length of y_train:', len(y_train))
print('length of X_test:', len(X_test))
print('length of y_test:', len(y_test))
"""
"""

for idx, item in enumerate(zip(X_train, y_train)):
    if idx == 5:
        break
    print('data:', item[0], ', target:', item[1])
"""

# データをPyTorchのテンソルに変換する。
import torch
"""
print("typeOf X_train = ", type(X_train))
print("typeOf y_train = ", type(y_train))
print("typeOf X_test = ", type(X_test))
print("typeOf y_test = ", type(y_test))
"""

X_train = torch.from_numpy(X_train).float()
y_train = torch.tensor([[float(x)] for x in y_train])  # ↓じゃだめなの？ tensor を使いたかっただけ？ 後でサイズが違うって言われた。
#y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.tensor([[float(x)] for x in y_test])  # ↓じゃだめなの？ tensor を使いたかっただけ？ 後でサイズが違うって言われた。
#y_test = torch.from_numpy(y_test).float()

"""
print("typeOf X_train = ", type(X_train))
print("typeOf y_train = ", type(y_train))
print("typeOf X_test = ", type(X_test))
print("typeOf y_test = ", type(y_test))
"""

# ------------------------------------------------------------------------------
# 2.ニューラルネットワークの定義
# ------------------------------------------------------------------------------
from torch import nn
import torch.nn.functional as F

INPUT_FEATURES = 4 # 入力層のノード数
HIDDEN         = 5 # 隠れ層のノード数
OUTPUT_CLASSES = 1 # 出力層のノード数

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # コンストラクタでは、層の定義だけを行う。
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN) # fc1 の fc は、おそらくファンクションかな？
        self.fc2 = nn.Linear(HIDDEN, OUTPUT_CLASSES) # fc2の出力を、そのままこのNetの出力にするから、出力層は無し。

    def forward(self, x):
        # forward で、層の繋がりを一連の処理として定める。
        x = self.fc1(x)  # 入力層の処理。この中で重みやバイアスの最適値を求めるの？
        x = F.sigmoid(x) # 活性化関数の使用
        x = self.fc2(x)  # 隠れ層の処理 （隠れ層って活性化関数は？）。この中で重みやバイアスの最適値を求めるの？
        return x

# 実際に、このネットのインスタンスを生成して呼び出してみる。
"""
net = Net()

outputs = net(X_train[0:3]) # とりあえず３行だけ食わせてみる。
print(outputs)
for idx in range(3):
    print("output = ", outputs[idx], ", target = ", y_train[idx])
"""
# ------------------------------------------------------------------------------
# 3.学習（訓練）と制度の検証
# ------------------------------------------------------------------------------
# だいたいの流れは下記

#    ニューラルネットワークにX_trainに格納したデータを入力する（112個）
#    損失関数を用いて、計算結果と正解ラベルとの誤差を計算する（計算結果は損失 と呼ばれる）
#    誤差逆伝播（バックプロパゲーション）や最適化と呼ばれる処理によって重みやバイアスを更新する
#    上記の処理を事前に定めた回数だけ繰り返す

net = Net()  # ニューラルネットワークのインスタンスを生成

criterion = nn.MSELoss()                # 損失関数（これも活性化関数みたいににいろんな種類あるんかな？）
optimizer = torch.optim.SGD(net.parameters(), lr=0.003)  # 最適化アルゴリズム

EPOCHS = 2000                           # 上と同じことを2000回繰り返す
for epoch in range(EPOCHS):
    optimizer.zero_grad()               # 重みとバイアスの更新で内部的に使用するデータをリセット
    outputs = net(X_train)              # 手順1：ニューラルネットワークにデータを入力
    loss = criterion(outputs, y_train)  # 手順2：正解ラベルとの比較
    loss.backward()                     # 手順3-1：誤差逆伝播
    optimizer.step()                    # 手順3-2：重みとバイアスの更新
    
    if epoch % 100 == 99:               # 100回繰り返すたびに損失を表示
        print(f'epoch: {epoch+1:4}, loss: {loss.data}')

print('training finished\n\n\n')

predict = (outputs + 0.5).int() # outputs を四捨五入
for idx, item in enumerate(zip(predict, y_train)):
    if idx == 5:
        break
    #print(item[0], '<--->', item[1])

compare = predict == y_train
#print(compare[0:5])
#print(compare.sum())

# 最後に評価
outputs = net(X_test)

predict = (outputs + 0.5).int()
compare = predict == y_test

print(f'correct: {compare.sum()} / {len(predict)}')
for value, label in zip(predict, y_test):
    print('predicted:', iris.target_names[value.item()], '<--->',iris.target_names[int(label.item())])