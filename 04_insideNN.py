# これはニューラルネットワークに入門するためのスタート
# ここからは下記を参考サイトとしていく。
# https://atmarkit.itmedia.co.jp/ait/series/18508/

# ここではまたiris（アヤメ）を利用
# 次の手順で進める。
#
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import nn
import torch
import torch.nn.functional as F

iris = load_iris()
#print(iris)
data = iris.data
target = iris.target


# iris.dataとiris.targetに含まれるデータをシャッフルして分割
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

# データをPyTorchのテンソルに変換する。

X_train = torch.from_numpy(X_train).float()
y_train = torch.tensor([[float(x)] for x in y_train]) 
X_test = torch.from_numpy(X_test).float()
y_test = torch.tensor([[float(x)] for x in y_test]) 


# ------------------------------------------------------------------------------
# 2.ニューラルネットワークの定義
# ------------------------------------------------------------------------------

INPUT_FEATURES = 4 # 入力層のノード数
HIDDEN         = 5 # 隠れ層のノード数
OUTPUT_CLASSES = 1 # 出力層のノード数

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # コンストラクタでは、層の定義だけを行う。
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN) # fc1 の fc は、おそらくファンクションかな？⇒full connectedやった。
        self.fc2 = nn.Linear(HIDDEN, OUTPUT_CLASSES) # fc2の出力を、そのままこのNetの出力にするから、出力層は無し。

    def forward(self, x):
        # forward で、層の繋がりを一連の処理として定める。
        x = self.fc1(x)  # 入力層の処理。この中で重みやバイアスの最適値を求めるの？
        x = F.sigmoid(x) # 活性化関数の使用
        x = self.fc2(x)  # 隠れ層の処理 （隠れ層って活性化関数は？）。この中で重みやバイアスの最適値を求めるの？
        return x

#
# ここで何が起きているかを確認していく
# 

net = Net()  # ニューラルネットワークのインスタンスを生成

# 重みとバイアスがある事を確認
print("weight")
print(net.fc1.weight)
print("bias")
print(net.fc1.bias)

# 明示的にfc1（入力層の処理　full connected = nn.Linear ）を呼び出して、
# 何が起きているか確認
x = X_train[0] # 訓練データの一つをとりだす。
print("x = ", x)

# 入力層の処理をしてみる。
result = net.fc1.forward(x) # 入力層の処理 forward してもおんなじ
print("**** result = ", result)

# 上記のfc1相当を自分で組んでみる。
w = net.fc1.weight
b = net.fc1.bias
x = X_train[0] # 訓練データの一つをとりだす。

o0 = w[0][0] * x[0] + w[0][1] * x[1] + w[0][2] * x[2] + w[0][3] * x[3] + b[0]
o1 = w[1][0] * x[0] + w[1][1] * x[1] + w[1][2] * x[2] + w[1][3] * x[3] + b[1]
o2 = w[2][0] * x[0] + w[2][1] * x[1] + w[2][2] * x[2] + w[2][3] * x[3] + b[2]
o3 = w[3][0] * x[0] + w[3][1] * x[1] + w[3][2] * x[2] + w[3][3] * x[3] + b[3]
o4 = w[4][0] * x[0] + w[4][1] * x[1] + w[4][2] * x[2] + w[4][3] * x[3] + b[4]
#print("**** datas = ", o0.data, o1.data, o2.data, o3.data, o4.data)

# これをPython 3.5では @ 演算子で計算してくれると。
o = w @ x + b
print("**** datas = ", o)

# 最後に、実際に１回訓練してみて、重みが更新される様子を見る。
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

print('before learning')
print('weight')
print(net.fc1.weight)
print('bias')
print(net.fc1.bias)

print('learn once')
outputs = net(X_train)                # １回学習
loss = criterion(outputs, y_train)    # そのロスを計算（結果＝outputsとy_trainの誤差） 
loss.backward()                       # 誤差逆伝播
optimizer.step()                      # 最適化
                                      # backward + step で、初めて更新されるな。

print('after learning')               # 確かにちょびっとだけど変わっている。
print('weight')
print(net.fc1.weight)
print('bias')
print(net.fc1.bias)

"""
# ------------------------------------------------------------------------------
# 3.学習（訓練）と制度の検証
# ------------------------------------------------------------------------------

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
#比較確認ように一覧表示
#for value, label in zip(predict, y_test):
#    print('predicted:', iris.target_names[value.item()], '<--->',iris.target_names[int(label.item())])
"""