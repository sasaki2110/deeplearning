# ここでは自作のライナークラスを作成する。

# まず基本形（ほぼNNクラスと同じ構造やなぁ）
import torch
from torch import nn
from math import sqrt

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()  # 基底クラスの初期化
        # 重みとバイアスを格納するインスタンス変数を定義

        self.in_features = in_features    # 入力値の数を保存
        self.out_features = out_features  # 出力値の数を保存

        # 重みを格納する行列の定義
        k = 1 / in_features
        weight = torch.empty(out_features, in_features).uniform_(-sqrt(k), sqrt(k))
        self.weight = nn.Parameter(weight)

        # バイアスを格納するベクトルの定義
        bias = torch.empty(out_features).uniform_(-k, k)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        # ある層からの入力xを受け取り、次の層への出力を計算する
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
# またまたiris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

X_train = torch.from_numpy(X_train).float()
y_train = torch.tensor([[float(x)] for x in y_train])
X_test = torch.from_numpy(X_test).float()
y_test = torch.tensor([[float(x)] for x in y_test])

# nn.Linnear と比較
INPUT_FEATURES = 4
HIDDEN = 5
OUTPUT_FEATURES = 1
"""
linear1 = torch.nn.Linear(INPUT_FEATURES, HIDDEN)
linear2 = MyLinear(INPUT_FEATURES, HIDDEN)

print('Linear class')
for param in linear1.parameters():
    print(param)

print('\nMyLinear class')
for param in linear2.parameters():
    print(param)
"""

# じゃあMyLinearクラスを使ってみる
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = MyLinear(INPUT_FEATURES, HIDDEN)
        self.fc2 = MyLinear(HIDDEN, OUTPUT_FEATURES)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

net = Net()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.003)

EPOCHS = 2000
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 99:
        print(f'epoch: {epoch+1:4}, loss: {loss.data}')

print('training finished')
predict = (outputs + 0.5).int()
compare = predict == y_train
print("学習の結果一致した件数は", compare.sum())