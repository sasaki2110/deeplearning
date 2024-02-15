# いよいよMINST手書き文字にチャレンジ。

# まずはMINST vision から手書き文字の訓練データ・テストデータを取得
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


# MINSTに含まれる画像（を構成する数値データ）をPyTorchで扱えるように変換してくれるらしい
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

BATCH_SIZE = 20

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# もう見慣れたNNの定義
class Net(torch.nn.Module):
    def __init__(self, INPUT_FEATURES, HIDDEN, OUTPUT_FEATURES):
        super().__init__()
        self.fc1 = torch.nn.Linear(INPUT_FEATURES, HIDDEN)
        self.fc2 = torch.nn.Linear(HIDDEN, OUTPUT_FEATURES)
        #self.softmax = torch.nn.Softmax(dim=1)   # softmaxがコメントアウトされている
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)           # 活性化がReLUになっとるな。
        x = self.fc2(x)
        #x = self.softmax(x)                      # softmaxがコメントアウトされている
        return x

INPUT_FEATURES = 28 * 28
HIDDEN = 64
OUTPUT_FEATURES = 10

net = Net(INPUT_FEATURES, HIDDEN, OUTPUT_FEATURES)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

EPOCHS = 2

for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0
    for count, item in enumerate(trainloader, 1):
        inputs, labels = item
        inputs = inputs.reshape(-1, 28 * 28)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if count % 500 == 0:
            print(f'#{epoch}, data: {count * 20}, running_loss: {running_loss / 500:1.3f}')
            running_loss = 0.0

print('Finished')

_, predicted = torch.max(outputs, 1)
print(predicted)
print(labels)

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = inputs.reshape(-1, 28 * 28)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += len(outputs)
        correct += (predicted == labels).sum().item()

print(f'correct: {correct}, accuracy: {correct} / {total} = {correct / total}')

"""
# データを少し見てみる
items = trainset.data[0]
print(f'image: {len(items)} x {len(items[0])}')
for item1 in items:            # 訓練データ１個目の、行を取得
    for item2 in item1:                   # その行の列を取得
        print(f'{item2.data:4}', end='')  # 行・列の値を順番に4桁に整形して表示
    print()                               # １行書き終わったら、改行

# １個のデータは 28 x 28 の画像データで、その画像データは 0〜255 の数値で表現されている。
# ちなみに正解ラベルは？？？？ dataではなく、trainset[0] ならunpackできるみたい。
    
# plotしようと（pngにするために、小細工を弄している。サイトのサンプルとは違うけど、まあ結果はOKやろう）
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

figure = plt.figure(figsize=(28, 28))
plt.axis("off")
plt.imshow(items, cmap="gray")
figure.savefig("sin.png")

image, label = trainset[0]
print(f'image: {len(image)} x {len(image[0])}')
print(f'label: {label}')

image = image.reshape(28, 28)
print()
for item1 in image:
    for item2 in item1:
        print(f' {float(item2.data):+1.2f} ', end='')
    print()
"""
