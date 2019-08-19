import torch.nn as nn

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
from transformer.Modules import utils, PositionalEncoding, Embeddings


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, pretrained_embeddings=None):

        super().__init__()

        n_position = len_max_seq + 1

        if pretrained_embeddings is None:
            #self.src_word_emb = nn.Embedding(
              #  n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
            self.src_word_emb = Embeddings(embedding_dim=d_model, padding_idx=Constants.PAD, vocab_size=n_src_vocab)
        else:
            self.src_word_emb = nn.Embedding.from_pretrained(
                pretrained_embeddings, padding_idx=Constants.PAD)

        '''self.position_enc = nn.Embedding.from_pretrained(
            utils.get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)'''

        self.position_enc = PositionalEncoding(d_model, len_max_seq)

        self.segment_enc = nn.Embedding(int(n_position / 2), d_word_vec, padding_idx=0)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, src_seg, return_attns=False):
        """
        First creates an input embedding from the seq, pos, and seg encodings.
        then runs the encoder layer for n_layers and returns the final vector

        Args:
            h_seq: Encodings for the words in the history
            h_pos: Positional encodings for the words in the history
            h_seg: Segment encodings for turns in the history
        Returns:
            enc_output: vector output from encoder
        """
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = utils.get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = utils.get_non_pad_mask(src_seq)
        emb = self.src_word_emb(src_seq)
        # -- Get input embeddings
        enc_output =  self.position_enc(emb) # + self.segment_enc(src_seg)

        # Nx encoder layer
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
