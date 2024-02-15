# ここからは下記を参考サイトとしていく。
# https://atmarkit.itmedia.co.jp/ait/series/18508/

# ニューラルネットワークの学習とは？

# シンプルに y = w * x で考える。
# 重み w の初期値を1.95、その正解ラベルの値を2.00とすることにする
# つまり「f(x) = 1.95 * x」という式が、「f(x) = 2.00 * x」と
# なっていく姿を観察する。
# x = 1 としてしまう。
# 最初はf(x)の出力は1.95だが、最終的に2.00になってほしいという事。

# 途中に実験的な話が混じってきて、今回の本筋がぶれたので、もう一度やり直し。

import torch

w = torch.tensor([[1.95]], requires_grad=True)  # 重みの初期値は1.95とする
t = torch.tensor([[2.0]])                       # 重みの正解は2.0
x = torch.tensor([1.0])                         # 関数への入力は1.0とする

def f(x):  # 関数f(x) = w * xの定義
    return w * x

# 損失関数の定義（平均二乗誤差）
criterion = torch.nn.MSELoss()
print("最初の重みを表示")
print(w)

# 最適化あるごり済みを指定。lr が学習率というらしい。
optimizer = torch.optim.SGD([w], lr=0.3)

# １回目の訓練
# まず勾配を初期化
print("---------１回目の学習--------------")
print(w.grad)
optimizer.zero_grad()
print(w.grad)

y = f(x)
print(y)
loss = criterion(y, t)
loss.backward()
print('updated w.grad:', w.grad)
optimizer.step()
print('updated w:', w)



# ２回目の訓練
# まず勾配を初期化

print("---------２回目の学習--------------")
print(w.grad)
optimizer.zero_grad()
print(w.grad)

y = f(x)
print(y)
loss = criterion(y, t)
loss.backward()
print('updated w.grad:', w.grad)
optimizer.step()
print('updated w:', w)


# ３回目の訓練
# まず勾配を初期化

print("---------３回目の学習--------------")
print(w.grad)
optimizer.zero_grad()
print(w.grad)

y = f(x)
print(y)
loss = criterion(y, t)
loss.backward()
print('updated w.grad:', w.grad)
optimizer.step()
print('updated w:', w)


