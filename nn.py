import torch
import torch.nn as nn
import numpy as np

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
        for str in activations:
            self.activations.append(get_activation(str))
        self.output_shape = output_shape
        if kernel_init == 'glorot_uniform':
            self.weight_init = weights_init_xavier
        else:
            self.weight_init = weights_init_normal
        self.layers = nn.ModuleList([nn.Linear(in_features=input_shape[0], out_features=self.units[0]), self.activations[0]])
        if dropout:
            self.layers.append(nn.Dropout(dropout[0]))
        for i in range(1, len(units)):
            self.layers.append(nn.Linear(in_features=units[i-1], out_features=units[i]))
            if activations[i]:
                self.layers.append(self.activations[i])
            if dropout:
                self.layers.append(nn.Dropout(dropout[i]))
        # self.fcc = nn.ParameterList(*layers)
        self.layers.apply(self.weight_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.output_shape:
            x = x.view(-1, *self.output_shape)
        return x


class SingleBlock(nn.Module):
    def __init__(self, in_features, out_features, activation, weight_init, batchnorm=True, dropout=None):
        super().__init__()
        layers = []
        layer = nn.Linear(in_features=in_features, out_features=out_features)
        weight_init(layer)
        layers.append(layer)
        layers.append(activation)
        if batchnorm:
            batchnorm_ = nn.BatchNorm2d(out_features)
            weight_init(batchnorm_)
            layers.append(batchnorm_)
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.ModuleList(layers)
        self.layers.apply(self.weight_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, filters, kernel_sizes, paddings, activations, poolings, kernel_init, input_shape=None, output_shape=None, dropouts=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shape = output_shape
        self.activations = []
        for str in activations:
            self.activations.append(get_activation(str))
        if kernel_init == nn.init.xavier_uniform_:
            self.weight_init = weights_init_xavier
        else:
            self.weight_init = weights_init_normal
        for i, (filter, ksize, padding, act, pooling) in enumerate(zip(filters, kernel_sizes, paddings, self.activations, poolings)):
            if i == 0 and input_shape:
                self.layers.append(nn.Conv2d(in_channels=input_shape, out_channels=filters[0], kernel_size=ksize, padding=padding))
                if act:
                    self.layers.append(act)
            else:
                self.layers.append(nn.Conv2d(in_channels=filters[i-1], out_channels=filters[i], kernel_size=ksize, padding=padding))
                if act:
                    self.layers.append(act)
            if dropouts and dropouts[i]:
                self.layers.append(nn.Dropout(dropouts[i]))
            if pooling:
                self.layers.append(nn.MaxPool2d(pooling))

        # self.modules = nn.Sequential(*self.layers)
        self.layers.apply(self.weight_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.output_shape:
            x = x.view(-1, self.output_shape)
        return x


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
        block_output = self.block(block_input)

        outputs = [input_vec, block_output]
        if self.concat_outputs:
            outputs = torch.cat(outputs, dim=1)
        return outputs


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





