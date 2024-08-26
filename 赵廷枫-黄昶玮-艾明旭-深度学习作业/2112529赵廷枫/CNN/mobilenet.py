import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def standard_conv_block(in_channel, out_channel, strid=1):  # 定义Strandard convolutional layer with batchnorm and ReLU
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, strid, 1, bias=False),  # conv
        nn.BatchNorm2d(out_channel),  # bn
        nn.ReLU()  # relu
    )
 
 
def depthwise_separable_conv_block(in_channel, out_channel,
                                   strid=1):  # 定义Depthwise Separable convolutions with Depthwise and Pointwise layers followed by batchnorm and ReLU
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, 3, strid, 1, groups=in_channel, bias=False),  # conv,使用与输入通道数相同组数的分组卷积实现Depthwise Convolution
        nn.BatchNorm2d(in_channel),  # bn
        nn.ReLU(),  # relu
        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),  # 1x1conv,Pointwise Convolution
        nn.BatchNorm2d(out_channel),  # bn
        nn.ReLU()  # relu
    )
 
 
class MobileNetV1(nn.Module):  # 定义MobileNet结构
    def __init__(self, num_classes=10):  # 初始化方法
        super(MobileNetV1, self).__init__()  # 继承初始化方法
 
        self.num_classes = num_classes  # 类别数量
        self.feature = nn.Sequential(  # 特征提取部分
            standard_conv_block(3, 4, strid=2),  # standard conv block,(n,3,32,32)-->(n,4,32,32)
            depthwise_separable_conv_block(4, 8,strid=2),  # depthwise separable conv block,(n,4,32,32)-->(n,8,16,16)
            depthwise_separable_conv_block(8, 8),  # depthwise separable conv block,(n,8,16,16)-->(n,8,16,16)
            depthwise_separable_conv_block(8, 16, strid=2),  # depthwise separable conv block,(n,8,16,16)-->(n,16,8,8)
            depthwise_separable_conv_block(16, 16),  # depthwise separable conv block,(n,16,8,8)-->(n,16,8,8)
            depthwise_separable_conv_block(16, 32, strid=2),  # depthwise separable conv block,(n,16,8,8)-->(n,16,4,4)
            depthwise_separable_conv_block(32, 64),  # depthwise separable conv block,(n,16,4,4)-->(n,16,2,2)
            depthwise_separable_conv_block(64, 128, strid=2),  # depthwise separable conv block,(n,16,2,2)-->(n,16,1,1)
            nn.AdaptiveAvgPool2d(1)  # avgpool,为方便后续转为特征向量,这里将avgpool放入特征提取部分,(n,1024,7,7)-->(n,1024,1,1)
        )
        self.classifier = nn.Sequential(  # 分类部分
            nn.Linear(128, self.num_classes),  # linear,(n,16)-->(n,num_classes)
            # nn.Softmax(dim=1)  # softmax
        )
 
    def forward(self, x):  # 前传函数
        x = self.feature(x)  # 特征提取，获得特征层,(n,16,1,1)
        x = torch.flatten(x, 1)  # 将三维特征层压缩至一维特征向量,(n,16,1,1)-->(n,16)
        return self.classifier(x)  # 分类，输出类别概率,(n,num_classes)
    
mobilenet=MobileNetV1()
print(mobilenet)




transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


optimizer = optim.Adam(mobilenet.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()

losses = []
accuracies = []
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        
        outputs = mobilenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 99:    # print every 2000 mini-batches
            average_loss = running_loss / 100
            accuracy = correct / total
            acc=accuracy*100
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {average_loss:.3f}, accuracy: {acc:.3f}%')
            losses.append(average_loss)
            accuracies.append(accuracy)
            running_loss = 0.0
            correct = 0
            total = 0

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Training Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('MobileNet training_metrics.png')  # 保存图像到当前文件夹
plt.show()

print('Finished Training')

# Test
dataiter = iter(testloader)
images, labels = next(dataiter)


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = mobilenet(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = mobilenet(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')