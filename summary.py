import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from thop import clever_format
from thop import profile

import pcode.create_dataset as create_dataset
from pcode.master_utils import do_validation
from pruneagent import PruneAgent

pruneagent = PruneAgent(10, 'cuda', minimal_ratio=0.0625)
checkpoint = torch.load('checkpoint_0.0001_0')
model = checkpoint['model']
retention_number = checkpoint['retention_number']
score = checkpoint['score']
pruneagent.get_original_filters_number(model)
pruneagent.score = score
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), 4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)
# 测试集
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)
trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
with torch.no_grad():
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            pruneagent.minimal_filter.append(int(pruneagent.minimal_ratio * len(pruneagent.score[name])))
pruneagent.minimal_filter.append(sum(pruneagent.minimal_filter))


def Accuracy(testloader, net, device):
    # 使用测试数据测试网络
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # 将输入和目标在每一步都送入GPU
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs[1], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    return 100.0 * correct / total


sum_flops = 0
sum_params = 0
for i, retention in enumerate(reversed(retention_number)):
    # for i in range(200):
    if i % 20 == 0:
        # retention = int(24 * (200 - i))
        pruneagent.master_model = copy.deepcopy(model)
        pruneagent.retention_number = retention
        pruneagent.dense_chs_idx = {}
        optimizer = torch.optim.SGD(pruneagent.master_model.parameters(), lr=0.1)
        with torch.no_grad():
            for name, param in pruneagent.master_model.named_parameters():
                if 'weight' in name:
                    if param.ndim == 1:
                        pruneagent.dense_chs_idx[name] = np.arange(param.shape[0])
                    if param.ndim == 2:
                        pruneagent.dense_chs_idx[name] = np.arange(param.shape[1])
                    if param.ndim == 4:
                        pruneagent.dense_chs_idx[name] = {'in_chs': np.arange(param.shape[1]),
                                                          'out_chs': np.arange(param.shape[0])}

        pruneagent.prune(pruneagent.master_model, optimizer, 0)
        pruneagent.master_model.unfreeze_bn()
        for image, target in trainloader:
            pruneagent.master_model(image.to('cuda'))
        acc = Accuracy(testloader, pruneagent.master_model, 'cuda')
        input = torch.randn(1, 3, 32, 32).to('cuda')
        flops, params = profile(pruneagent.master_model, inputs=(input,), verbose=False)
        if i == 0:
            origin_flops = flops
            origin_params = params
        sum_params += params
        sum_flops += flops
        flops_rate = flops / origin_flops
        params_rate = params / origin_params
        flops, params = clever_format([flops, params], "%.3f")
        print(
            f'model: {i}, retention: {retention}, flops: {flops}, flops rate: {flops_rate}, params: {params}, params rate: {params_rate}')

# print(f'global flops: {sum_flops/(origin_flops*10)}, global params: {sum_params/(origin_params*10)}.')
