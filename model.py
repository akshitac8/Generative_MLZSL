import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        layer_sizes = opt.encoder_layer_sizes
        latent_size = opt.attSize
        in_c = layer_sizes[0] + latent_size
        self.fc1 = nn.Linear(in_c, layer_sizes[-1])
        self.fc3 = nn.Linear(layer_sizes[-1], latent_size*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear_means = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var = nn.Linear(latent_size*2, latent_size)
        self.apply(weights_init)

    def forward(self, x, att=None):
        if att is not None: x = torch.cat((x, att), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc3(x))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

## EARLY FUSION ##
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        input_size = opt.attSize * 2
        self.fc1 = nn.Linear(input_size, opt.decoder_layer_sizes[0])
        self.fc2 = nn.Linear(opt.decoder_layer_sizes[0], opt.decoder_layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, z, att):
        z = torch.cat((z, att), dim=-1)
        x1 = self.lrelu(self.fc1(z))
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        return x1

# Generator body used for late and hybrid fusion
class body_generator(nn.Module):
    def __init__(self, opt, att_size):
        super(body_generator, self).__init__()
        layer_sizes = opt.decoder_layer_sizes
        input_size = att_size * 2
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        x = torch.cat((x, att), 1)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return x

## EARLY FUSION ##
class LATE_FUSION(nn.Module):
    def __init__(self, opt):
        super(LATE_FUSION, self).__init__()
        self.late_fusion = body_generator(opt, opt.attSize)
        self.resSize = opt.resSize
    def forward(self, noise, att): #feat-> [BSX4096] #att->list (Nx312) with len=BS
        late_out = torch.zeros(len(att),self.resSize).cuda()
        idx_count = torch.zeros(len(att)).cuda()
        for j in range(att.size(1)):
            idx = [i for i in range(len(att)) if att[i,j].abs().sum() > 0]
            idx_count[idx] += 1
            late_out[idx] += torch.sigmoid(self.late_fusion(noise[idx],att[idx,j].cuda()))
        late_out = late_out/idx_count.unsqueeze(1).clamp(min=1)
        final_output = late_out
        return final_output

## HYBRID FUSION SELF ATTENTION ###
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear  = nn.Linear(d_model, d_model, bias=False)
        self.v_linear  = nn.Linear(d_model, d_model, bias=False)
        self.k_linear  = nn.Linear(d_model, d_model, bias=False)
        self.out       = nn.Linear(d_model, d_model, bias=False)
        self.dropout_1   = nn.Dropout(dropout)
        self.dropout_2   = nn.Dropout(dropout)

    def forward(self, q, k, v):
        bs = q.size(0)
        residual = q
        # perform linear operation and split into h heads ## transpose to get dimensions bs * h * sl * d_model
        k = self.k_linear(k).view(bs, self.h, -1, self.d_k)
        q = self.q_linear(q).view(bs, self.h, -1, self.d_k)
        v = self.v_linear(v).view(bs, self.h, -1, self.d_k)
        scores = attention(q, k, v, self.d_k, self.dropout_1)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.dropout_2(self.out(concat))
        output += residual
        return output

def attention(q, k, v, d_k, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = scores.masked_fill(scores == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores) 
    output = torch.matmul(scores, v)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.num_ff = num_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.apply(weights_init)

    def forward(self, x):
        residual = x
        x = self.linear_2(F.leaky_relu(self.linear_1(x), 0.2, True))
        x += residual
        return x

class MultiheadAttentionlayer(nn.Module):
    def __init__(self, heads, d_model, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.feedforward_net = FeedForward(d_model, d_ff)

    def forward(self, x, leaky_relu=False):
        x = self.self_attn(x, x, x)
        x = self.feedforward_net(x, leaky_relu)
        return x

class HYBRID_FUSION_ATTENTION(nn.Module):
    def __init__(self, opt):
        super(HYBRID_FUSION_ATTENTION, self).__init__()
        self.resSize, self.N, self.hiddensize, late_heads  = opt.resSize, opt.N, opt.hiddensize, 8
        self.early_fusion = body_generator(opt, opt.attSize)
        self.late_fusion = body_generator(opt, opt.attSize)
        self.attn = MultiheadAttentionlayer(late_heads, self.resSize, self.hiddensize)

    def forward(self, noise, att, avg_att):
        late_out = torch.zeros(len(att),self.resSize).cuda()
        idx_count = torch.zeros(len(att)).cuda()  
        for j in range(att.size(1)):
            idx = [i for i in range(len(att)) if att[i,j].abs().sum() > 0]
            idx_count[idx] += 1
            late_out[idx] += torch.sigmoid(self.late_fusion(noise[idx],att[idx,j].cuda()))        
        late_out = late_out/idx_count.unsqueeze(1).clamp(min=1)
        early_out = torch.sigmoid(self.early_fusion(noise, avg_att))
        temp_out = torch.stack((late_out,early_out),1)
        out = self.attn(temp_out, leaky_relu=self.leaky_relu)
        out = torch.sigmoid(torch.mean(out,1))
        return out

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att=None):
        if att is not None: x = torch.cat((x, att), 1)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return x