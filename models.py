import torch
import torch.nn as nn
import torch.nn.functional as F

class EncodeLayer_G(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=True, dropout=0.0):
        super(EncodeLayer_G, self).__init__()
        modules = nn.Sequential()
        modules.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1))
        if activate:
            modules.add_module("act", nn.LeakyReLU(0.2))
            modules.add_module("bn", nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01))
            
        if dropout > 0.0:
            modules.add_module("dropout", nn.Dropout(dropout))
        self.block = modules
    
    def forward(self, x):
        return self.block(x)

class DecodeLayer_G(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale, activate=True, dropout=0.0):
        super(DecodeLayer_G, self).__init__()
        modules = nn.Sequential()
        modules.add_module("up", nn.Upsample(scale))
        modules.add_module("deconv", nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=1))
        if activate:
            modules.add_module("act", nn.LeakyReLU(0.2))
            modules.add_module("bn", nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01))
            
        if dropout > 0.0:
            modules.add_module("dropout", nn.Dropout(dropout))
        self.block = modules
    
    def forward(self, x):
        return self.block(x)
    
class EncoderBlock_G(nn.Module):
    def __init__(self, pre_channels, in_channels, out_channels, num_layers, dropout_rate=0.0):
        super(EncoderBlock_G, self).__init__()
        
        self.pre_conv = EncodeLayer_G(pre_channels, pre_channels, 3, 2, False)
        self.conv0 = EncodeLayer_G(in_channels + pre_channels, out_channels, 3, 1)
        total_channels = in_channels + out_channels
        self.conv1 = EncodeLayer_G(total_channels, out_channels, 3, 1)
        total_channels += out_channels
        self.conv2 = EncodeLayer_G(total_channels, out_channels, 3, 1)
        total_channels += out_channels
        self.conv3 = EncodeLayer_G(total_channels, out_channels, 3, 2, dropout=dropout_rate)
        
    def forward(self, inp):
        pre_input, x = inp
        pre_input = self.pre_conv(pre_input)
        h = self.conv0(torch.cat((x, pre_input), dim=1))
        
        all_outputs = [x, h]
        in_features = torch.cat(all_outputs, dim=1)
        h = self.conv1(in_features)
        all_outputs.append(h)
        
        in_features = torch.cat(all_outputs, dim=1)
        h = self.conv2(in_features)
        all_outputs.append(h)
        
        in_features = torch.cat(all_outputs, dim=1)
        h = self.conv3(in_features)
        all_outputs.append(h)
        return all_outputs[-2], all_outputs[-1]
    
class DecoderBlock_G(nn.Module):
    def __init__(self, pre_channels, in_channels, out_channels, num_layers, curr_size, scale=None, dropout_rate=0.0):
        super(DecoderBlock_G, self).__init__()
        self.should_upscale = scale is not None
        self.should_pre_conv = pre_channels > 0
        
        total_channels = pre_channels + in_channels
        if self.should_pre_conv:
            self.pre_conv_t0 = DecodeLayer_G(pre_channels, pre_channels, 3, curr_size, False)
        self.conv0 = EncodeLayer_G(total_channels, out_channels, 3, 1)
        total_channels += out_channels
        
        if self.should_pre_conv:
            self.pre_conv_t1 = DecodeLayer_G(pre_channels, pre_channels, 3, curr_size, False)
        self.conv1 = EncodeLayer_G(total_channels, out_channels, 3, 1)
        total_channels += out_channels
        
        if self.should_pre_conv:
            self.pre_conv_t2 = DecodeLayer_G(pre_channels, pre_channels, 3, curr_size, False)
        self.conv2 = EncodeLayer_G(total_channels, out_channels, 3, 1)
        total_channels += out_channels
        
        if self.should_upscale:
            total_channels = total_channels - pre_channels
            self.conv_t3 = DecodeLayer_G(total_channels, out_channels, 3, scale, True, dropout_rate)
            
    def forward(self, inp):
        pre_input, x = inp
        all_outputs = [x]
        
        curr_input = all_outputs[-1]
        if self.should_pre_conv:
            pre_conv_output = self.pre_conv_t0(pre_input)
            curr_input = torch.cat([curr_input, pre_conv_output], dim=1)
        input_features = torch.cat([curr_input]+all_outputs[:-1], dim=1)
        h = self.conv0(input_features)
        all_outputs.append(h)
        
        curr_input = all_outputs[-1]
        if self.should_pre_conv:
            pre_conv_output = self.pre_conv_t1(pre_input)
            curr_input = torch.cat([curr_input, pre_conv_output], dim=1)
        input_features = torch.cat([curr_input]+all_outputs[:-1], dim=1)
        h = self.conv1(input_features)
        all_outputs.append(h)
        
        curr_input = all_outputs[-1]
        if self.should_pre_conv:
            pre_conv_output = self.pre_conv_t2(pre_input)
            curr_input = torch.cat([curr_input, pre_conv_output], dim=1)
        input_features = torch.cat([curr_input]+all_outputs[:-1], dim=1)
        h = self.conv2(input_features)
        all_outputs.append(h)
        
        if self.should_upscale:
            input_features = torch.cat(all_outputs, dim=1)
            h = self.conv_t3(input_features)
            all_outputs.append(h)
        return all_outputs[-2], all_outputs[-1]
        
        
class Generator(nn.Module):
    def __init__(self, channels, curr_size=[2, 4, 8, 16, 32], layer_size=[64, 64, 128, 128], dropout_rate=0.0):
        super(Generator, self).__init__()
        self.channels = channels
        self.curr_size = curr_size
        self.layer_size = layer_size
        num_inner = 3
        
        self.encoder0 = EncodeLayer_G(channels, 64, 3, 2)
        self.encoder1 = EncoderBlock_G(channels, 64, 64, num_inner, dropout_rate)
        self.encoder2 = EncoderBlock_G(64, 64, 128, num_inner, dropout_rate)
        self.encoder3 = EncoderBlock_G(128, 128, 128, num_inner, dropout_rate)
        
        self.projection = Linear_projection(curr_size=2, z_dim=100)
        
        in_channels = 128 + 8
        self.decoder0 = DecoderBlock_G(0, in_channels, 128, num_inner, self.curr_size[0], 4, dropout_rate)
        in_channels = 128 + 128 + 4
        self.decoder1 = DecoderBlock_G(128, in_channels, 128, num_inner, self.curr_size[1], 8, dropout_rate)
        in_channels = 128 + 64 + 2
        self.decoder2 = DecoderBlock_G(128, in_channels, 64, num_inner, self.curr_size[2], 16, dropout_rate)
        in_channels = 64 + 64
        self.decoder3 = DecoderBlock_G(64, in_channels, 64, num_inner, self.curr_size[3], 32, dropout_rate)
        in_channels = channels + 64
        self.decoder4 = DecoderBlock_G(64, in_channels, 64, num_inner, self.curr_size[4], None, dropout_rate)
        
        self.final_conv0 = EncodeLayer_G(64, 64, 3, 1, activate=True)
        self.final_conv1 = EncodeLayer_G(64, 64, 3, 1, activate=True)
        self.final_conv2 = EncodeLayer_G(64, 3, 3, 1, activate=False)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x, z):
        all_outputs = [x, self.encoder0(x)]
        
        h = [x, self.encoder0(x)]
        h = self.encoder1(h)
        all_outputs.append(h[1])
        
        h = self.encoder2(h)
        all_outputs.append(h[1])
        
        h = self.encoder3(h)
        all_outputs.append(h[1])
        
        pre_input, curr_input = None, h[1]
        code1, code2, code3 = self.projection(z)
        code1 = code1.view(code1.size(0), -1, self.curr_size[0], self.curr_size[0])
        code2 = code2.view(code2.size(0), -1, self.curr_size[1], self.curr_size[1])
        code3 = code3.view(code3.size(0), -1, self.curr_size[2], self.curr_size[2])
        
        curr_input = torch.cat([code1, curr_input], dim=1)
        pre_input, curr_input = self.decoder0([pre_input, curr_input])
        
        curr_input = torch.cat([curr_input, all_outputs[-2]], dim=1)
        curr_input = torch.cat([code2, curr_input], dim=1)
        pre_input, curr_input = self.decoder1([pre_input, curr_input])
        
        curr_input = torch.cat([curr_input, all_outputs[-3]], dim=1)
        curr_input = torch.cat([code3, curr_input], dim=1)
        pre_input, curr_input = self.decoder2([pre_input, curr_input])
        
        curr_input = torch.cat([curr_input, all_outputs[-4]], dim=1)
        pre_input, curr_input = self.decoder3([pre_input, curr_input])
        curr_input = torch.cat([curr_input, all_outputs[-5]], dim=1)
        pre_input, curr_input = self.decoder4([pre_input, curr_input])
        
        curr_input = self.final_conv0(curr_input)
        curr_input = self.final_conv1(curr_input)
        curr_input = self.final_conv2(curr_input)
        return self.tanh(curr_input)
          
class Linear_projection(nn.Module):
    def __init__(self, curr_size=None, z_dim=100):
        super(Linear_projection, self).__init__()
        n_noise_filter = 8
        self.proj1 = nn.Linear(z_dim, curr_size*curr_size*n_noise_filter)
        curr_size = curr_size * 2
        n_noise_filter = n_noise_filter // 2
        self.proj2 = nn.Linear(z_dim, curr_size*curr_size*n_noise_filter)
        curr_size = curr_size * 2
        n_noise_filter = n_noise_filter // 2
        self.proj3 = nn.Linear(z_dim, curr_size*curr_size*n_noise_filter)
        
    def forward(self, x):
        return self.proj1(x), self.proj2(x), self.proj3(x)
        
        
####################################################
# Discriminator
####################################################
class _LayerNorm(nn.Module):
    def __init__(self, num_features, img_size):
        """
        Normalizes over the entire image and scales + weights for each feature
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            (num_features, img_size, img_size), elementwise_affine=False, eps=1e-12
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features).float().unsqueeze(-1).unsqueeze(-1),
            requires_grad=True,
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features).float().unsqueeze(-1).unsqueeze(-1),
            requires_grad=True,
        )

    def forward(self, x):
        out = self.layer_norm(x)
        out = out * self.weight + self.bias
        return out

class ConvLayer_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, out_size=None, activate=True, dropout=0.0):
        super(ConvLayer_D, self).__init__()
        modules = nn.Sequential()
        modules.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1))
        if activate:
            modules.add_module("act", nn.LeakyReLU(0.2))
            modules.add_module("ln", _LayerNorm(out_channels, out_size))
            
        if dropout > 0.0:
            modules.add_module("dropout", nn.Dropout(dropout))
        self.block = modules
    
    def forward(self, x):
        return self.block(x)
    
class EncoderBlock_D(nn.Module):
    def __init__(self, pre_channels, in_channels, out_channels, num_layers, out_size, dropout_rate=0.0):
        super(EncoderBlock_D, self).__init__()
        self.pre_conv = ConvLayer_D(pre_channels, pre_channels, 3, 2, activate=False)
        self.conv0 = ConvLayer_D(in_channels + pre_channels, out_channels, 3, 1, out_size=out_size)
        
        total_channels = in_channels + out_channels
        self.conv1 = ConvLayer_D(total_channels, out_channels, 3, 1, out_size=out_size)
        total_channels += out_channels
        self.conv2 = ConvLayer_D(total_channels, out_channels, 3, 1, out_size=out_size)
        total_channels += out_channels
        self.conv3 = ConvLayer_D(total_channels, out_channels, 3, 1, out_size=out_size)
        total_channels += out_channels
        self.conv4 = ConvLayer_D(total_channels, out_channels, 3, 1, out_size=out_size)
        total_channels += out_channels
        self.conv5 = ConvLayer_D(total_channels, out_channels, 3, 2, out_size=(out_size+1)//2, dropout=dropout_rate)
                
    def forward(self, inp):
        pre_input, x = inp
        pre_input = self.pre_conv(pre_input)
        h = self.conv0(torch.cat([x, pre_input], dim=1))
        
        all_outputs = [x, h]
        input_features = torch.cat([all_outputs[-1], all_outputs[-2]] + all_outputs[:-2], dim=1)
        h = self.conv1(input_features)
        all_outputs.append(h)
        
        input_features = torch.cat([all_outputs[-1], all_outputs[-2]] + all_outputs[:-2], dim=1)
        h = self.conv2(input_features)
        all_outputs.append(h)
        
        input_features = torch.cat([all_outputs[-1], all_outputs[-2]] + all_outputs[:-2], dim=1)
        h = self.conv3(input_features)
        all_outputs.append(h)
        
        input_features = torch.cat([all_outputs[-1], all_outputs[-2]] + all_outputs[:-2], dim=1)
        h = self.conv4(input_features)
        all_outputs.append(h)
        
        input_features = torch.cat([all_outputs[-1], all_outputs[-2]] + all_outputs[:-2], dim=1)
        h = self.conv5(input_features)
        all_outputs.append(h)
        return all_outputs[-2], all_outputs[-1]
    
class Discriminator(nn.Module):
    def __init__(self, channels, dropout_rate=0.0, z_dim=100):
        super(Discriminator, self).__init__()
        num_inner = 5
        self.curr_size = [32, 16, 8, 4]
        self.encoder0 = ConvLayer_D(channels, 64, 3, 2, out_size=self.curr_size[1])
        self.encoder1 = EncoderBlock_D(channels, 64, 64, num_inner, out_size=self.curr_size[1], dropout_rate=dropout_rate)
        self.encoder2 = EncoderBlock_D(64, 64, 128, num_inner, out_size=self.curr_size[2], dropout_rate=dropout_rate)
        self.encoder3 = EncoderBlock_D(128, 128, 128, num_inner, out_size=self.curr_size[3], dropout_rate=dropout_rate)
        
        self.dense1 = nn.Linear(128, 1024)
        self.dense2 = nn.Linear(128*2**2+1024, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = [x, self.encoder0(x)]
        h = self.encoder1(h)
        h = self.encoder2(h)
        h = self.encoder3(h)
        h = h[1]
        
        h_mean = h.mean([2, 3])
        h_flat = torch.flatten(h, start_dim=1)
        
        h = self.dense1(h_mean)
        h = self.lrelu(h)
        h = self.dense2(torch.cat([h, h_flat], dim=1))
        return h