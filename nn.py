import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

custom_objects = {}

activations = {'relu': nn.ReLU(), 'elu': nn.ELU()}


class CustomActivation(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)


def get_activation(str):
    if str in activations:
        return activations[str]
    elif str:
        act = eval(str)
        act = CustomActivation(act)
        return act
    return None


class FC(nn.Module):       #fully connected convolutional block
    def __init__(self, units, activations, input_shape, output_shape=None, dropout=None, kernel_init='normal'):
        super().__init__()
        self.dropout = dropout
        self.units = units
        self.activations = []
        named_modules = []
        for str in activations:
            self.activations.append(get_activation(str))
        self.output_shape = output_shape
        named_modules.append(('Linear_FC_0', nn.Linear(in_features=input_shape[0], out_features=self.units[0])))
        named_modules.append(('Activation_FC_0', self.activations[0]))
        if dropout:
            named_modules.append(('Dropout_FC_0', nn.Dropout(dropout[0])))
        for i in range(1, len(units)):
            named_modules.append((f'Linear_FC_{i}', nn.Linear(in_features=units[i-1], out_features=units[i])))
            if activations[i]:
                named_modules.append((f'Activation_FC_{i}', self.activations[i]))
            if dropout:
                named_modules.append((f'Dropout_FC_{i}', nn.Dropout(dropout[i])))
        # self.fcc = nn.ParameterList(*layers)
        self.layers = nn.Sequential(OrderedDict(named_modules))
        if kernel_init == 'glorot_uniform':
            self.weight_init = weights_init_xavier
        else:
            self.weight_init = weights_init_normal
        self.layers.apply(self.weight_init)

    def forward(self, x):
        if len(x) == 2:
            return [self.layers(x[0]), x[1]]
        x = self.layers(x)
        if self.output_shape:
            x = x.view(-1, *self.output_shape)
        return [x]


class SingleBlock(nn.Module):
    def __init__(self, in_features, out_features, activation, weight_init, batchnorm=True, dropout=None):
        super().__init__()
        layers = []
        layer = nn.Linear(in_features=in_features, out_features=out_features)
        layers.append(layer)
        layers.append(activation)
        if weight_init == 'glorot_uniform':
            self.weight_init = weights_init_normal
        else:
            self.weight_init = weights_init_xavier
        if batchnorm:
            batchnorm_ = nn.BatchNorm2d(out_features)
            layers.append(batchnorm_)
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.weight_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


activation_reg = {}
def get_activation_by_name(name):
    def hook(model, input, output):
        activation_reg[name] = output
    return hook


class Regressor(nn.Module):

    def __init__(self, in_channels, kernel_size, filter, stride=None, padding=None):
        super(Regressor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=filter,
                               kernel_size=kernel_size,
                               stride=2) #add padding support if needed
        # self.conv2 = nn.Conv2d(in_channels=filter,
        #                        out_channels=2 * filter,
        #                        kernel_size=kernel_size)
        self.activation = get_activation('relu')
        self.fc = nn.Linear(in_features=filter, out_features=1)
        # self.dropout = nn.Dropout(0.05)
        self.weight_init = weights_init_xavier
        self.conv1.apply(self.weight_init)
        # self.conv2.apply(self.weight_init)
        self.fc.apply(self.weight_init)

    def forward(self, x): #--> (128, 64, 4, 4)
        x = self.conv1(x) #--> (128, 64, 1, 1)
        x = self.activation(x)
        # x = self.dropout(x)
        x = x.view(-1, 128)
        # x = self.conv2(x) #--> (128, 64, 1, 1)
        # x = self.activation(x)
        x = self.fc(x)
        return self.activation(x)


class ConvBlock(nn.Module):
    def __init__(self, filters, kernel_sizes, paddings, activations, poolings, kernel_init, input_shape=None, output_shape=None, dropouts=None):
        super().__init__()
        self.output_shape = output_shape
        self.activations = []
        named_modules = []
        self.reg = Regressor(in_channels=64, kernel_size=3, filter=128)
        for str in activations:
            self.activations.append(get_activation(str))
        for i, (filter, ksize, padding, act, pooling) in enumerate(zip(filters, kernel_sizes, paddings, self.activations, poolings)):
            if i == 0 and input_shape:
                named_modules.append((f"Conv2d_{i}", nn.Conv2d(in_channels=input_shape, out_channels=filters[0], kernel_size=ksize, padding=padding)))
                if act:
                    named_modules.append((f"Activation_{i}", act))
            else:
                named_modules.append((f"Conv2d_{i}", nn.Conv2d(in_channels=filters[i-1], out_channels=filters[i], kernel_size=ksize, padding=padding)))
                if act:
                    named_modules.append((f"Activation_{i}", act))
            if dropouts and dropouts[i]:
                named_modules.append((f"Dropout_{i}", nn.Dropout(dropouts[i])))
            if pooling:
                named_modules.append((f"MaxPool_{i}", nn.MaxPool2d(pooling)))
        # self.modules = nn.Sequential(*self.layers)
        self.layers = nn.Sequential(OrderedDict(named_modules))
        self.layers.MaxPool_2.register_forward_hook(get_activation_by_name('MaxPool_2'))
        if kernel_init == nn.init.xavier_uniform_:
            self.weight_init = weights_init_xavier
        else:
            self.weight_init = weights_init_normal
        self.layers.apply(self.weight_init)
        self.reg.apply(self.weight_init)

    def forward(self, x):
        x = self.layers(x)
        if self.output_shape:
            x = x.view(-1, self.output_shape)
        reg_output = self.reg(activation_reg['MaxPool_2'])
        return x, reg_output


class VImgConcat(torch.nn.Module):
    def __init__(self, vector_shape, img_shape, block, vector_bypass=False, concat_outputs=True):
        super().__init__()
        self.vector_shape = tuple(vector_shape)
        self.img_shape = tuple(img_shape)
        self.block = block
        self.vector_bypass = vector_bypass
        self.concat_outputs = concat_outputs

    def forward(self, xx):
        input_vec, input_img = xx[0], xx[1]
        block_input = input_img
        if len(self.img_shape) == 2:
            block_input = block_input.view(-1, *(self.img_shape + (1, )))
        reshaped_vec = torch.tile(torch.reshape(input_vec, (-1, *((1, 1) + self.vector_shape))), dims=(1, *self.img_shape[:2], 1))
        block_input = torch.cat((block_input, reshaped_vec), dim=-1)

        block_input = torch.permute(block_input, (0, 3, 1, 2))
        block_output, reg_output = self.block(block_input)

        outputs = [input_vec, reg_output, block_output]
        if self.concat_outputs:
            outputs = torch.cat(outputs, dim=1)
        return [outputs, reg_output]


def build_block(block_type, arguments):
    if block_type == 'fully_connected':
        block = FC(**arguments)
    elif block_type == 'conv':
        block = ConvBlock(**arguments)
    elif block_type == 'connect':
        inner_block = build_block(**arguments['block'])
        arguments['block'] = inner_block
        block = VImgConcat(**arguments)
    else:
        raise (NotImplementedError(block_type))

    return block


class BuildModel(nn.Module):
    def __init__(self, block_descriptions):
        super().__init__()
        self.models = nn.ModuleList([build_block(**desc) for desc in block_descriptions])

    def forward(self, x):
        for model in self.models:
            x = model(x)
        return x


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
