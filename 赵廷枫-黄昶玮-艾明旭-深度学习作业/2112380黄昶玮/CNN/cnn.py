import torch
import torchvision
import torchvision.transforms as transforms


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




import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 4, 6, 28, 28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
#         x = self.conv1(x)
#         print("output shape of conv1:", x.size())
#         x = F.relu(x)
        
#         x = self.pool(x)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# net = Net().to('cuda')
# print(net)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=100):
        super(ResNet,self).__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self.make_layer(block,64,layers[0])
        self.layer2=self.make_layer(block,128,layers[1],2)
        self.layer3=self.make_layer(block,256,layers[2],2)
        self.layer4=self.make_layer(block,512,layers[3],2)
        self.avg_pool=nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
                )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn(out)
        out=self.relu(out)
        out=self.maxpool(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.avg_pool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out
    
def resnet(num_classes=10):
    return ResNet(BasicBlock,[2,2,2,2],num_classes)



# net = ResNet(BasicBlock,[2,2,2,2],10).to('cuda')
# print(net)


import torch.nn as nn
import torch.nn.functional as F
class DenseBasic(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBasic, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=(1, 1)),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x):
        out = torch.cat((x, self.layer(x)), dim=1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        block = []
        for i in range(6):
            block.append(DenseBasic(in_channels, growth_rate))
            in_channels += growth_rate
        self.denseblock = nn.Sequential(*block)

    def forward(self, x):
        return self.denseblock(x)


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.denseblock1 = DenseBlock(32, 32)
        self.bn2 = nn.BatchNorm2d(224)
        self.conv1 = nn.Conv2d(224, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.avg1 = nn.AvgPool2d(2, stride=2)
        self.denseblock2 = DenseBlock(64, 64)
        self.fc1 = nn.Linear(7168, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = self.denseblock1(x)
        x = self.bn2(x)
        x = self.conv1(x)
        x = self.avg1(x)
        x = self.denseblock2(x)
        batch_size, channels, w, h = x.shape
        x = x.reshape(batch_size, channels * w * h)
        x = self.fc1(x)
        return x
    
# Densenet = DenseNet().to('cuda')
# print(Densenet)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MobleNetV1(nn.Module):
    def __init__(self, num_classes):
        super(MobleNetV1, self).__init__()
        self.conv1 = self._conv_st(3, 32, 2)
        self.conv_dw1 = self._conv_dw(32, 64, 1)
        self.conv_dw2 = self._conv_dw(64, 128, 2)
        self.conv_dw3 = self._conv_dw(128, 128, 1)
        self.conv_dw4 = self._conv_dw(128, 256, 2)
        self.conv_dw5 = self._conv_dw(256, 256, 1)
        self.conv_dw6 = self._conv_dw(256, 512, 2)
        self.conv_dw_x5 = self._conv_x5(512, 512, 5)
        self.conv_dw7 = self._conv_dw(512, 1024, 2)
        self.conv_dw8 = self._conv_dw(1024, 1024, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_dw3(x)
        x = self.conv_dw4(x)
        x = self.conv_dw5(x)
        x = self.conv_dw6(x)
        x = self.conv_dw_x5(x)
        x = self.conv_dw7(x)
        x = self.conv_dw8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

    def _conv_x5(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def _conv_st(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _conv_dw(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
net=MobleNetV1(10).to('cuda')
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def test(model,testloader,criterion,device):
    model.eval()
    test_loss=0
    correct=0
    top5_correct=0
    total=0
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(testloader):
            data,target=data.to(device),target.to(device)
            output=model(data)
            loss=criterion(output,target)
            test_loss+=loss.item()
            _,predicted=output.max(1)
            total+=target.size(0)
            correct+=predicted.eq(target).sum().item()
            _,top5_predicted=output.topk(5,1)
            top5_correct+=top5_predicted.eq(target.view(-1,1).expand_as(top5_predicted)).sum().item()
    return test_loss/(batch_idx+1),100.*correct/total,100.*top5_correct/total


test_losses=[]
test_accs=[]
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    test_loss,test_acc,test_top5_acc=test(net,testloader,criterion,'cuda')
    test_losses.append(test_loss)
    test_accs.append(test_acc)

print('Finished Training')

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,11), test_losses)
plt.title('test_loss')

plt.figure(figsize=(5,3))
plt.plot(np.arange(1,11), test_accs)
plt.title('test_acc')