import torch.nn as nn
import torch.nn as nn

import transformer.Constants as Constants
from transformer.Layers import DecoderLayer
from transformer.Modules import utils, PositionalEncoding, Embeddings


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, pretrained_embeddings=None):

        super().__init__()
        n_position = len_max_seq + 1

        if pretrained_embeddings is None:
            '''self.tgt_word_emb = nn.Embedding(
                n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)'''
            self.tgt_word_emb = Embeddings(embedding_dim=d_model, padding_idx=Constants.PAD, vocab_size=n_tgt_vocab)
        else:
            self.tgt_word_emb = nn.Embedding.from_pretrained(
                pretrained_embeddings, padding_idx=Constants.PAD)

        '''self.position_enc = nn.Embedding.from_pretrained(
            utils.get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)'''
        self.position_enc = PositionalEncoding(d_model, len_max_seq)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        """
        Starts by getting the imput embedding from the target seq, and pos
        encodings. Then runs the decoder.

        Args:
            tgt_seq: Encodings for the words in the target response
            tgt_pos: Positional encodings for the words in the target response
            src_seq: Encodings for the words in the history
            enc_output: Output from the Encoder
        Returns:
            sec_output: vector outputs from decoder, one for each word in the response

        """
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = utils.get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = utils.get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = utils.get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = utils.get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        emb = self.tgt_word_emb(tgt_seq)

        # -- Forward
        dec_output =  self.position_enc(emb)

        # Nx decoder layer
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,
