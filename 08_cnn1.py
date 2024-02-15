# CNN （畳み込み　＆　プーリング）

import torch
import torchvision
import torchvision.transforms as transforms

# とりあえず、MINSTの読み込み
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
BATCH_SIZE = 20
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)     # 畳み込み1 
                                                  # 1:チャネル数（grayだから1） 
                                                  # 6:出力チャネルの数＝カーネル数 
                                                  # 5:カーネルサイズ（５×５）
        
        self.pool = torch.nn.MaxPool2d(2, 2)      # プーリング(2×2のサイズでプーリングする)

        self.conv2 = torch.nn.Conv2d(6, 16, 5)    # 畳み込み2
                                                  # 6:入力チャネルの数＝conv1の出力チャネルの数
                                                  # 16:出力チャネルの数＝カーネル数
                                                  # 5:カーネルサイズ（５×５）
        
        self.fc1 = torch.nn.Linear(16 * 16, 64)   # 全結合層1（入力：16*16、出力：64）
        self.fc2 = torch.nn.Linear(64, 10)        # 全結合層2（入力：64、出力：10）
    def forward(self, x):
        x = self.conv1(x)                         # 畳み込み1
        x = torch.nn.functional.relu(x)           # 活性化関数（ReLU）
        x = self.pool(x)                          # プーリング
        x = self.conv2(x)                         # 畳み込み2
        x = self.pool(x)                          # プーリング
        x = x.reshape(-1, 16 * 16)                # １次元にして
        x = self.fc1(x)                           # 全結合層1（入力：16*16、出力：64）
        x = torch.nn.functional.relu(x)           # 活性化関数（ReLU）
        x = self.fc2(x)                           # 全結合層2（入力：64、出力：10）
        return x

# 学習してみる
import torch.optim as optim

net = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

EPOCHS = 2

for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0
    for count, item in enumerate(trainloader, 1):
        inputs, labels = item

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

# 評価してみる
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += len(outputs)
        correct += (predicted == labels).sum().item()

print(f'correct: {correct}, accuracy: {correct} / {total} = {correct / total}')
