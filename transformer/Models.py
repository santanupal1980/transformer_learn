''' Define the Transformer model '''
import torch.nn as nn

from transformer.decoder import Decoder
from transformer.encoder import Encoder


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq_enc, len_max_seq_dec,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True,
            pretrained_embeddings=None):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq_enc,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, pretrained_embeddings=pretrained_embeddings)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq_dec,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, pretrained_embeddings=pretrained_embeddings)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
                "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, src_seg, tgt_seq, tgt_pos):
        """
        Takes in the input features for the history and response, and returns a prediction.

        First encodes the history, and then decodes it before mapping the output to the vocabulary

        Args:
            src_seq: Encodings for the words in the history
            src_pos: Positional encodings for the words in the history
            src_seg: Segment encodings for turns in the history
            tgt_seq: Encodings for the words in the target response
            tgt_pos: Positional encodings for the words in the target response
        Returns:
            Outputs: Vector of probabilities for each word in the vocabulary, for each word in the response
        """

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos, src_seg)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        outputs = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return outputs.view(-1, outputs.size(2))
