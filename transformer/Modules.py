import numpy as np
import torch.nn.functional as F

import torch
from torch import nn
from torch.autograd import Function
import math

from transformer import Constants


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        gets the querys keys, and values for each attention head.
        the queries and keys are multiplied, and this is scaled, masked,
         softmaxed, and dropouted to get the weights
         these weights are applied to the values via matrix multiplication
        Args:
            q: Query
            k: Key
            v: Value
            mask:
        Returns:
        """

        # MatMul
        #attn = torch.matmul(q, k.transpose(1, 2))
        attn = torch.matmul(q, k.transpose(2, 3))
        # Scale
        attn = attn / self.temperature

        # Mask
        if mask is not None:
            #attn = attn.masked_fill(mask, -np.inf)
            attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # softmax/dropout
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # Matmul
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.head_size = d_model // n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)


        #nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        #self.fc = nn.Linear(n_head * d_v, d_model)
        #nn.init.xavier_normal_(self.fc.weight)
        self.output_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        first passes the querys, keys, and values through linear layers to get
        n_head inputs for scaled dot-product attention
        then preforms scaled Dot-product attention and applies layer normalization
        before returning the output
        Args:
            q: Query
            k: Key
            v: Value
            mask:
        Returns:
            Output: output from multihead attention
        """

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        batch_size, len_q, _ = q.size()
        batch_size, len_k, _ = k.size()
        batch_size, len_v, _ = v.size()
        residual = q


        ''' # Linear
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        '''

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, self.n_head, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        #mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        # Scaled Dot-Product Attention
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_head * self.head_size)

        #output = output.view(n_head, batch_size, len_q, d_v)
        #output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)  # b x lq x (n*dv)

        #output = self.dropout(self.fc(output))

        output = self.output_layer(output)

        # Add and Norm
        #output = self.layer_norm(output.transpose(1, 2) + residual)

        return output, attn

class utils:

    def get_non_pad_mask(seq):
        assert seq.dim() == 2
        return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

    def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
        ''' Sinusoid position encoding table '''

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.FloatTensor(sinusoid_table)

    def get_attn_key_pad_mask(seq_k, seq_q):
        ''' For masking out the padding part of key sequence. '''

        # Expand to fit the shape of key query attention matrix.
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(Constants.PAD)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

        return padding_mask

    def get_subsequent_mask(seq):
        ''' For masking out the subsequent info. '''

        sz_b, len_s = seq.size()
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

        return subsequent_mask





class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''



    def __init__(self, d_hid, d_in, dropout=0.1):
        super().__init__()
        self.L_1 = nn.Linear(d_hid, d_in)  # position-wise
        self.L_2 = nn.Linear(d_in, d_hid)  # position-wise
        self.layer_norm = nn.LayerNorm(d_hid, eps=1e-06)
        self.dropout = nn.Dropout(dropout)



    def forward(self, x):
        """
        just a feed forward linear layer that is used after attention in the
        encoder and decoder
        Args:
            x: input
        Returns:
        """
        # feed forward
        residual = x
        output = self.layer_norm(x)
        #output = output.transpose(1, 2)
        output = self.L_1(output)
        output = self.L_2(F.relu(output))
        #output = output.transpose(1, 2)
        output = self.dropout(output)

        # Add and norm
        output = output + residual
        return output
