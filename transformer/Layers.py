''' Define the Layers '''
import torch.nn as nn

from transformer.Modules import MultiHeadAttention, PositionwiseFeedForward, SelfAttn, CrossAttn


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = SelfAttn(
            n_head, d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        """
        First performs self attention on the input. then the result is passed
        through a feed forward network to get the output
        Args:
            enc_input: vector input
        Returns:
            enc_output: vector output from encoder layer
        """
        # Multi-Head Attention (w/ Add and Norm)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        # TODO CHECK add&norm or Norm&add
        enc_output = enc_output + enc_input
        enc_output = self.layer_norm(enc_output)

        # Feed forward (w/ Add and Norm)
        enc_output_ff = self.pos_ffn(enc_output)
        enc_output = enc_output + enc_output_ff
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = SelfAttn(n_head, d_model, dropout=dropout)
        self.enc_dec_attn = CrossAttn(n_head, d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        """
        First performs masked self attention on input.
        Then preforms attention
        where the query is the output from the previous layer, and the keys
        and values is the encoder output
        finally, the result is passed through a feed forward network to get
        the output
        Args:
            dec_input: input to the decoder
            enc_output: output from encoder
        Returns:
            dec_output: output from decoder
        """
        # Masked Multi-Head Attention (w/ Add and Norm)
        dec_output, dec_slf_attn = self.slf_attn(dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        # Multi-Head Attention (w/ Add and Norm)
        dec_output, dec_enc_attn = self.enc_dec_attn(dec_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        # TODO CHECK add&norm or Norm&add
        dec_output = dec_output + dec_input
        dec_output = self.layer_norm(dec_output)

        # Feed forward (w/ Add and Norm)
        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
