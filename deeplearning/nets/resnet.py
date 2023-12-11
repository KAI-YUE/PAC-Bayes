import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled, flatten, Tensor
from torchvision.models import resnet

from deeplearning.nets_utils import EmbeddingRecorder

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_32x32(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, record_embedding: bool = False,
                 no_grad: bool = False, penultimate: bool = False):
        super().__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(channel, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad
        self.penultimate = penultimate

    # by DM
    ###
    def params(self):
        for name, param in self.named_params(self):
            yield param
    
    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p
    
    def update_params(self, lr_inner, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                # NOTE:
                #print(type(param_t)) #This is Parameter
                #print(type(grad)) # But, this is Tensor!
                tmp = param_t - lr_inner * grad
                #print(tmp)
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    #if first_order:
                    #    grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param): #name = curr_mod_layer
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            # NOTE:
            #setattr(curr_mod, name, param) # Need to convert all the Parameter into Tensor
            curr_mod._parameters[name] = param # Parameter -> Tensor
    ###

    def get_last_layer(self):
        return self.linear

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out_cnn = out.view(out.size(0), -1)
            out = self.embedding_recorder(out_cnn)
            out = self.linear(out_cnn)
        if self.penultimate == False:
            return out
        else:
            return out, out_cnn


class ResNet_224x224(resnet.ResNet):
    def __init__(self, block, layers, channel: int, num_classes: int, record_embedding: bool = False,
                 no_grad: bool = False, **kwargs):
        super().__init__(block, layers, **kwargs)
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        if channel != 3:
            self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes != 1000:
            self.fc = nn.Linear(self.fc.in_features, num_classes)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        with set_grad_enabled(not self.no_grad):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = flatten(x, 1)
            x = self.embedding_recorder(x)
            x = self.fc(x)

        return x


def ResNet(arch: str, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
           pretrained: bool = False, penultimate: bool = False):
    arch = arch.lower()
    if pretrained:
        if arch == "resnet18":
            net = ResNet_224x224(resnet.BasicBlock, [2, 2, 2, 2], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_224x224(resnet.BasicBlock, [3, 4, 6, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 6, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 23, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_224x224(resnet.Bottleneck, [3, 8, 36, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url(resnet.model_urls[arch], progress=True)
        net.load_state_dict(state_dict)

        if channel != 3:
            net.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes != 1000:
            net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif im_size[0] == 224 and im_size[1] == 224:
        if arch == "resnet18":
            net = ResNet_224x224(resnet.BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_224x224(resnet.BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_224x224(resnet.Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
    elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        if arch == "resnet18":
            net = ResNet_32x32(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        elif arch == "resnet34":
            net = ResNet_32x32(BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        elif arch == "resnet50":
            net = ResNet_32x32(Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        elif arch == "resnet101":
            net = ResNet_32x32(Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        elif arch == "resnet152":
            net = ResNet_32x32(Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad, penultimate=penultimate)
        else:
            raise ValueError("Model architecture not found.")
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


def ResNet18(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False, penultimate: bool = False):
    return ResNet("resnet18", channel, num_classes, im_size, record_embedding, no_grad, pretrained, penultimate)


def ResNet34(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet("resnet34", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def ResNet50(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet("resnet50", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def ResNet101(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet("resnet101", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def ResNet152(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet("resnet152", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
