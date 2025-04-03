from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim
from transformers import AutoModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
# from models.hugging_gpt2.GPT2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
import copy

class SelfAttention1(nn.Module):
    def __init__(self, feature_dim, target_dim, num_auxiliary):
        super(SelfAttention1, self).__init__()
        self.target_dim = target_dim  
        self.num_auxiliary = num_auxiliary
        self.scale = 1.0 / np.sqrt(feature_dim)
        self.M_st = nn.Parameter(torch.randn(1, num_auxiliary))

        self.query = nn.Linear(1, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x_target,x_auxiliary,x_auxiliary_pooled,x_auxiliary_pooled2):

        q1 = self.query(x_target)
        k1 = self.key(x_auxiliary_pooled) 
        v1 = self.value(x_auxiliary_pooled2)
        attention_scores = torch.matmul(q1.transpose(-2, -1), k1) * self.scale
        M_st_row = self.M_st[:, self.target_dim].unsqueeze(0).unsqueeze(-1).expand(-1, -1, k1.size(-1))
        attention_scores = attention_scores * M_st_row
        attention_scores_reduced = torch.sum(attention_scores, dim=1, keepdim=True)
        attention_weights = F.softmax(attention_scores_reduced, dim=-1)
        weighted_values = torch.matmul(attention_weights, v1)
        M_st_expanded = self.M_st.unsqueeze(0).expand(weighted_values.size(0), -1, -1)
        final_weights = weighted_values + M_st_expanded
        final_weights_expanded = final_weights.expand(-1, x_auxiliary.size(1), -1)
        x_original_weighted = x_auxiliary * final_weights_expanded
        return x_original_weighted


class Adapter(nn.Module):
    def __init__(self, in_feat, hid_dim, skip=True):
        super().__init__()
        self.D_fc1 = nn.Linear(in_feat, hid_dim)
        self.D_fc2 = nn.Linear(hid_dim, in_feat)
        self.act = nn.GELU()
        self.skip = skip
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x_intermediate = self.D_fc1(x)
        x_intermediate = self.act(x_intermediate)
        x_intermediate = self.D_fc2(x_intermediate)
        x_output = self.drop(x_intermediate)
        if self.skip:
            x_output = x_output + x
        print("Output shape:", x_output.shape)
        return x_output
        

class SpectModule(nn.Module):
    def __init__(self, freq_len, adapter_len):
        super().__init__()
        self.adapter_len = adapter_len
        self.weight_r = nn.Parameter(torch.rand(freq_len, adapter_len//2))
        self.weight_i = nn.Parameter(torch.rand(freq_len, adapter_len//2))
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        B, M, N, P = x.shape
        x = rearrange(x, 'b m n p -> b m p n')
        x_ft = torch.fft.rfft(x, dim=-1)

        x_real = x_ft.real
        x_imag = x_ft.imag
        x_real = torch.einsum("bmpn, nd->bmpd", x_real, self.weight_r)
        x_imag = torch.einsum("bmpn, nd->bmpd", x_imag, self.weight_i)
        x_ft = torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))

        res = torch.fft.irfft(x_ft, dim=-1, n=self.adapter_len)
        res = rearrange(res, 'b m p n -> b m n p')

        return self.drop(res)


class SpectBlock(nn.Module):
    def __init__(self, in_feat, freq_len, low_rank=8, adapter_len=8):
        super().__init__()
        self.ln_1 = nn.LayerNorm(in_feat)
        self.ln_2 = nn.LayerNorm(in_feat)
        self.attn = SpectModule(freq_len//2+1, adapter_len)
    
    def forward(self, x):
        x = self.attn(self.ln_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class FFT_adapter(nn.Module):
    def __init__(self, n_layer, in_feat, seq_len):
        super().__init__()
        self.blocks = nn.ModuleList([SpectBlock(in_feat, seq_len) for i in range(n_layer)])

    def forward(self, x):
        res_list = []
        for i, block in enumerate(self.blocks):
            res_list.append(block(x))
        
        return res_list


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.pred_len = configs.pred_len
        self.d_ff = configs.d_ff
        self.gpt_layers = configs.gpt_layers
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        if self.stride > 1 or self.patch_size > 1:
            self.patch_num += 1
        
        # self.gpt2 = GPT2Model.from_pretrained('/your model road/',output_attentions=True, output_hidden_states=True)
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model    
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        for i in range(configs.gpt_layers):
            self.gpt2.h[i].scale = configs.scale
            self.gpt2.h[i].attn.scale = configs.scale
            if configs.T_type == 1:
                self.gpt2.h[i].T_adapter = Adapter(configs.d_model, configs.adapter_dim, skip=False)
                self.gpt2.h[i].T_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.patch_num, 1))
            if configs.C_type == 1:
                self.gpt2.h[i].C_adapter = Adapter(configs.d_model, configs.adapter_dim, skip=False)
                self.gpt2.h[i].C_num = configs.enc_in
                self.gpt2.h[i].C_adapter_gate = torch.nn.Parameter(torch.zeros(1, configs.enc_in, 1))


        self.fft_adapter = FFT_adapter(configs.spect_adapter_layer, configs.enc_in, self.patch_num)
        self.adapter_in_layer = nn.ModuleList([nn.Linear(configs.patch_len, configs.d_model) for i in range(configs.adapter_layer)])
        self.in_layer = nn.Linear(configs.patch_len, configs.d_model)

        self.proj_layer = nn.Linear(configs.d_model, self.d_ff)
        self.out_layer = nn.Linear(self.d_ff * self.patch_num, configs.pred_len)
        self.out_layer = nn.Linear(self.d_ff * self.patch_num, configs.pred_len)
        self.self_attentions1 = nn.ModuleList([
            SelfAttention1(feature_dim=28, target_dim=i, num_auxiliary=28) for i in range(4)
        ])

    
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'adapter' in name:
                param.requires_grad = True
            elif 'ln' in name:
                param.requires_grad = True
            elif 'wpe' in name:
                param.requires_grad = False
            else:
                param.requires_grad = False

        params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        params = sum(p.numel() for p in self.in_layer.parameters() if p.requires_grad)
        params = sum(p.numel() for p in self.out_layer.parameters() if p.requires_grad)
        params = sum(p.numel() for p in self.fft_adapter.parameters() if p.requires_grad)

    def forward(self, x, *args, **kwargs):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        model_idx = kwargs['model_idx'] 
        self_attention1 = self.self_attentions1[model_idx]
        x_target = x[:, :, model_idx+3:model_idx+4]  
        x_target = F.adaptive_avg_pool1d(x_target.transpose(1, 2), 1).transpose(1, 2)
        x_auxiliary = x[:, :, 7:]  
        x_auxiliary_pooled = F.adaptive_avg_pool1d(x_auxiliary.transpose(1, 2), 1).transpose(1, 2)
        x_auxiliary_pooled2 = F.adaptive_avg_pool1d(x_auxiliary.transpose(1, 2), 28).transpose(1, 2)
        # Get dynamic weights from attention
        dynamic_weights = self_attention1(x_target, x_auxiliary, x_auxiliary_pooled, x_auxiliary_pooled2)
        # Handle static weights
        static_weights = kwargs.get('static_weights', None)
        if static_weights is None:
            static_weights = torch.zeros_like(dynamic_weights, device=dynamic_weights.device)
        else:
            static_weights = torch.tensor(static_weights, device=dynamic_weights.device)
            static_weights = static_weights.view(1, 1, -1)  
            static_weights = static_weights.expand(
                dynamic_weights.size(0),  
                dynamic_weights.size(1),  
                -1                       
            )
        final_weights = (dynamic_weights + static_weights) / 2
        weighted_values = final_weights

        x4 = x[:, :, :7]  # Target feature

        x = torch.cat((x4, weighted_values), dim=2)
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        fft_adapter_list = self.fft_adapter(x) 
        for idx, tensor in enumerate(fft_adapter_list):
            print(f"Shape of tensor {idx}: {tensor.shape}")


        adapters = []
        for i in range(self.gpt_layers - len(fft_adapter_list)):
            adapters.append(None)
        for i in range(len(fft_adapter_list)):
            fft_adapter_list[i] = self.adapter_in_layer[i](fft_adapter_list[i])
            fft_adapter_list[i] = rearrange(fft_adapter_list[i], 'b m n p -> (b m) n p')
            adapters.append(fft_adapter_list[i])
        
        x = rearrange(x, 'b m n p -> (b m) n p')
        outputs = self.in_layer(x)
        outputs = self.gpt2(inputs_embeds=outputs, adapters=adapters).last_hidden_state
        outputs = self.proj_layer(outputs)
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means
        outputs = outputs[:, :,3:4]

        return outputs
