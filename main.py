'''简单的SNN图像分类任务，来源https://snntorch.readthedocs.io/en/latest/tutorials/zh-cn/tutorial_5_cn.html
   MNIST数据集为0-9手写数字识别'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML
from snntorch import utils
import numpy as np

commands={"coding":"rate", #脉冲编码方式，rate/latency
}

# 1. 数据加载和预处理
transform = transforms.Compose([  #数据预处理操作。这些操作的作用是将原始图像数据转换为适合模型训练的格式
            transforms.Resize((28,28)), #图像处理为28*28，在MNIST中已经是了
            transforms.Grayscale(), #转为灰度图，在MNIST中已经是了
            transforms.ToTensor(), # 将图像转换为张量
            transforms.Normalize((0.1307,), (0.3081,))]) # 对每个像素值标准化处理

# 训练参数
batch_size=128
num_classes = 10  # 0-9共10个数字

# torch变量float
dtype = torch.float

# 加载 MNIST 数据集并剪裁
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform) #MNIST提前定义了训练与测试，以train参数区分
# subset = 10 #对数据集剪裁到1/10大小，可以不执行
# mnist_train = utils.data_subset(mnist_train, subset)
# print(f"The size of train_dataset is {len(mnist_train)}") #f前缀表示这是一个格式化字符串，{}按表达式执行，比str.format简洁还快
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True) #shuffle：打乱顺序

mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform) #测试集
# mnist_test = utils.data_subset(mnist_test, subset)
# print(f"The size of test_dataset is {len(mnist_test)}") 
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# 2. 定义 SNN 网络
class SNN(nn.Module):
    def __init__(self, num_steps):
        super(SNN, self).__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(28 * 28, 512)  # 输入层到隐藏层
        self.lif1 = snn.Leaky(beta=0.9) #不指定替代激活函数时默认以反正切函数为surrogate
        self.fc2 = nn.Linear(512, 10)  # 隐藏层到输出层
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):
        mem1 = self.lif1.init_leaky() # 初始化膜电位，权重不变
        mem2 = self.lif2.init_leaky()
        spk_rec = [] #记录输出层脉冲
        mem_rec = [] #记录输出层电位
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1) #神经元本质就是激活函数，用来给fc权重的电流转成脉冲
            cur2 = self.fc2(spk1) #对脉冲取权重
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2) #将每个时间步的输出脉冲和膜电位存储起来，以便后续分析或用于损失函数的计算。
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0) #转为张量，并按时间步轴堆叠

# 3.训练设置
# 按时间步累计脉冲，比较最高脉冲序号与目标值

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1)) # view将输入数据 data 重塑为 [batch_size, input_size] 的形状。-1 表示自动计算该维度的大小
    _, idx = output.sum(dim=0).max(1) #在时间维度上对脉冲总数求和作为速率脉冲总数，对输出维度求最大值（值与序号），值不需要所以仅保留序号
    acc = np.mean((targets == idx).detach().cpu().numpy()) #targets == idx：比较预测idx和标签targets，返回一个布尔张量

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%") #输出准确率

def train_printer( #训练结果输出器
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

# 4. 初始化网络和优化器
device = torch.device("cuda")
num_steps = 25  # SNN 的时间步数
net = SNN(num_steps=num_steps).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.9, 0.999))
loss = nn.CrossEntropyLoss()

# 5. 数据集脉冲编码，训练的时候没用到
if commands["coding"]=="rate":
    num_steps=num_steps #每个像素点表示为多长的脉冲序列，序列越长精度越高，但效率越低
    for data_it, targets_it in train_loader:
        # Spiking Data
        spike_data = spikegen.rate(data_it, num_steps=num_steps)
        # print(spike_data.size()) # 输入数据的结构是 [num_steps x batch_size x input dimensions]

if commands["coding"]=="latency":
    num_steps=num_steps
    for data_it, targets_it in train_loader:
        # Spiking Data
        spike_data = spikegen.latency(data_it,num_steps, normalize=False, linear=False, tau=5, threshold=0.01)
        #ssntorch中的latency编码同样按照RC电路原理，输入特征转为rc电路电流，因此有snn神经元参数
        #normalize后所有信号一定会在timestep里触发;不用linear就默认为对数延迟；tau是rc参数，threshold是膜电位阈值
        # print(spike_data.size()) # 输入数据的结构是 [num_steps x batch_size x input dimensions]

# 6. 训练 SNN 网络
        
num_epochs = 10
loss_hist = []
test_loss_hist = []
counter = 0

for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader) #在每次迭代中，for data, targets in train_batch 会从迭代器中获取一个批次的数据和目标标签。

    # 单批次训练
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # 前向传播
        net.train() #将模型设置为训练模式，启用batchnorm和dropout
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets) 
            #权重不仅影响当前时间步，还会影响之前的时间步，采用类似BPTT的时间步损失函数累加方法
            #nn.CrossEntropyLoss内部会将标签索引值转化为独热编码，且将网络输出值进行softmax，不需要外部操作
            #膜电位高≈脉冲多≈分类概率强

        # 反向传播
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # 记录loss画图用
        loss_hist.append(loss_val.item())

        # 测试集
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer(
                    data, targets, epoch,
                    counter, iter_counter,
                    loss_hist, test_loss_hist,
                    test_data, test_targets)
            counter += 1
            iter_counter +=1

# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()


