import numpy as np
import torch.nn.functional as F

import torch
from torch import nn
from torch.autograd import Function
from math import sqrt

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
        #attn = attn / self.temperature

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

class SparseNormer(nn.Module):

    # dim: dimension to normalize

    def __init__(self, dim=-1, ieps=1e-32):
        super(SparseNormer, self).__init__()

        self.dim = dim
        self.bias = nn.Parameter(torch.zeros(1))
        self.act = nn.ReLU(inplace=True)
        self.ieps = ieps

    def forward(self, x):
        _tmp = self.act(x + self.bias)
        _tmp = _tmp * _tmp

        # fix zero-devision in case all elements in _tmp are 0.
        return _tmp / (_tmp.sum(self.dim, keepdim=True) + self.ieps)

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, sparsenorm = False):
        super().__init__()

        self.attn_dim = d_model // n_head
        self.hsize = self.attn_dim * n_head
        self.n_head = n_head

        self.d_k = d_k
        self.d_v = d_v

        self.qs = nn.Linear(d_model, self.hsize)
        self.ks = nn.Linear(d_model, self.hsize)
        self.vs = nn.Linear(d_model, self.hsize)

        #self.outer = nn.Linear(self.hsize, osize, bias=enable_bias)

        self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

        #self.layer_norm = nn.LayerNorm(d_model)

        self.output_layer = nn.Linear(self.hsize, d_model)
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

        adim = self.attn_dim
        residual = q


        ''' # Linear

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        '''

        Q = self.qs(q).view(batch_size, len_q, n_head, adim).transpose(1, 2)
        K = self.ks(k).view(batch_size, len_k, n_head, adim).transpose(1, 2)
        V = self.vs(v).view(batch_size, len_v, n_head, adim).transpose(1, 2)

        # compute scores
        scores = torch.div(torch.matmul(Q, K.transpose(2, 3)), sqrt(adim))

        if mask is not None:
            scores.masked_fill_(torch.unsqueeze(mask, 1).expand_as(scores), -1e32)

        scores = self.normer(scores)

        if self.dropout is not None:
            scores = self.dropout(scores)



        output = torch.matmul(scores, V).transpose(1, 2).contiguous()


        output = self.output_layer(output.view(batch_size, len_q, self.hsize))

        # Add and Norm
        #output = self.layer_norm(output.transpose(1, 2) + residual)

        return output, scores

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


# Accelerated MultiHeadAttn for self attention, use when Q == K == V
class SelfAttn(nn.Module):

    # isize: input dimension
    # hsize: hidden dimension
    # osize: output size of this layer
    # num_head: number of heads
    # dropout: dropout probability
    # sparsenorm: using sparse normer or standard softmax

    def __init__(self, input_size, hid_size, out_size, num_head=8, dropout=0.0, enable_bias=False, sparsenorm=False):

        super(SelfAttn, self).__init__()

        self.attn_dim = hid_size // num_head
        self.hid_size = self.attn_dim * num_head
        self.num_head = num_head

        self.adaptor = nn.Linear(input_size, self.hid_size * 3, bias=enable_bias)

        self.outer = nn.Linear(self.hid_size, out_size, bias=enable_bias)

        # self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
        self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

        self.drop = nn.Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

    # iQ: query (bsize, num_query, vsize)
    # mask (bsize, num_query, seql)
    # iK: key/value (bsize, seql, vsize), in case key != query, for efficient decoding

    def forward(self, iQ, mask=None, iK=None):

        bsize, nquery, _ = iQ.size()
        nheads = self.num_head
        adim = self.attn_dim

        # real_iQ: MultiHead iQ (bsize, num_query, vsize) => (bsize, nheads, nquery, adim)
        # real_iK: MultiHead iK (bsize, nquery, vsize) => (bsize, nheads, nquery, adim)
        # real_iV: MultiHead iV (bsize, nquery, vsize) => (bsize, nheads, nquery, adim)

        if iK is None:

            _out = self.adaptor(iQ)

            Q, K, V = _out.narrow(-1, 0, self.hid_size).contiguous().view(bsize, nquery, nheads,adim).transpose(1,2), \
                      _out.narrow(-1, self.hid_size, self.hid_size).contiguous().view(bsize, nquery, nheads, adim).transpose(1, 2), \
                       _out.narrow(-1, self.hid_size + self.hid_size, self.hid_size).contiguous().view(bsize, nquery, nheads, adim).transpose(1, 2)
        else:

            real_iQ, _out = F.linear(iQ, self.adaptor.weight.narrow(0, 0, self.hsize),
                                          self.adaptor.bias.narrow(0, 0,
                                                                   self.hsize) if self.adaptor.bias else None).view(
                bsize, nquery, nheads, adim).transpose(1, 2), F.linear(iK,
                                                                            self.adaptor.weight.narrow(0, self.hsize,
                                                                                                       self.hsize + self.hsize),
                                                                            self.adaptor.bias.narrow(0, self.hsize,
                                                                                                     self.hsize + self.hsize) if self.adaptor.bias else None)

            seql = iK.size(1)

            real_iK, real_iV = _out.narrow(-1, 0, self.hsize).contiguous().view(bsize, seql, nheads, adim).transpose(1,
                                                                                                                     2), _out.narrow(
                -1, self.hsize, self.hsize).contiguous().view(bsize, seql, nheads, adim).transpose(1, 2)

        # scores (bsize, nheads, nquery, adim) * (bsize, nheads, nquery, adim)' => (bsize, nheads, nquery, nquery)

        scores = torch.div(torch.matmul(real_iQ, real_iK.transpose(2, 3)), sqrt(adim))

        if mask is not None:
            scores.masked_fill_(torch.unsqueeze(mask, 1).expand_as(scores), -1e32)

        scores = self.normer(scores)

        if self.drop is not None:
            scores = self.drop(scores)

        # oMA: output of MultiHeadAttention T((bsize, nheads, nquery, nquery) * (bsize, nheads, nquery, adim)) => (bsize, nquery, nheads, adim)

        oMA = torch.matmul(scores, real_iV).transpose(1, 2).contiguous()

        # output of this layer (bsize, nquery, nheads, adim) => (bsize, nquery, osize)

        return self.outer(oMA.view(bsize, nquery, self.hsize))


# Accelerated MultiHeadAttn for cross attention, use when K == V
class CrossAttn(nn.Module):

    # isize: input dimension
    # hsize: hidden dimension
    # osize: output size of this layer
    # num_head: number of heads
    # dropout: dropout probability
    # sparsenorm: using sparse normer or standard softmax

    def __init__(self, isize, hsize, osize, num_head=8, dropout=0.0, enable_bias=False, sparsenorm=False):

        super(CrossAttn, self).__init__()

        self.attn_dim = hsize // num_head
        self.hsize = self.attn_dim * num_head
        self.num_head = num_head

        self.query_adaptor = nn.Linear(isize, self.hsize, bias=enable_bias)
        self.kv_adaptor = nn.Linear(isize, self.hsize * 2, bias=enable_bias)

        self.outer = nn.Linear(self.hsize, osize, bias=enable_bias)

        # self.normer = MHSparseNormer(num_head, dim=-1) if sparsenorm else nn.Softmax(dim=-1)
        self.normer = SparseNormer(dim=-1) if sparsenorm else nn.Softmax(dim=-1)

        self.drop = nn.Dropout(dropout, inplace=sparsenorm) if dropout > 0.0 else None

    # iQ: query (bsize, num_query, vsize)
    # iK: keys (bsize, seql, vsize)
    # mask (bsize, num_query, seql)

    def forward(self, iQ, iK, mask=None):

        bsize, nquery, _ = iQ.size()
        seql = iK.size(1)
        nheads = self.num_head
        adim = self.attn_dim

        # real_iQ: MultiHead iQ (bsize, num_query, vsize) => (bsize, nheads, nquery, adim)
        # real_iK: MultiHead iK (bsize, seql, vsize) => (bsize, nheads, seql, adim)
        # real_iV: MultiHead iV (bsize, seql, vsize) => (bsize, nheads, seql, adim)

        real_iQ, _out = self.query_adaptor(iQ).view(bsize, nquery, nheads, adim).transpose(1, 2), self.kv_adaptor(iK)

        real_iK, real_iV = _out.narrow(-1, 0, self.hsize).contiguous().view(bsize, seql, nheads, adim).transpose(1,
                                                                                                                 2), _out.narrow(
            -1, self.hsize, self.hsize).contiguous().view(bsize, seql, nheads, adim).transpose(1, 2)

        # scores (bsize, nheads, nquery, adim) * (bsize, nheads, seql, adim)' => (bsize, nheads, nquery, seql)

        scores = torch.div(torch.matmul(real_iQ, real_iK.transpose(2, 3)), sqrt(adim))

        if mask is not None:
            scores.masked_fill_(torch.unsqueeze(mask, 1).expand_as(scores), -1e32)

        scores = self.normer(scores)

        if self.drop is not None:
            scores = self.drop(scores)

        # oMA: output of MultiHeadAttention T((bsize, nheads, nquery, seql) * (bsize, nheads, seql, adim)) => (bsize, nquery, nheads, adim)

        oMA = torch.matmul(scores, real_iV).transpose(1, 2).contiguous()

        # output of this layer (bsize, nquery, nheads, adim) => (bsize, nquery, osize)

        return self.outer(oMA.view(bsize, nquery, self.hsize))


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
