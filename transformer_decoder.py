import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward_network import FeedForwardNetwork


class TransformerDecoder(nn.Module):
    ''' Transformer decoder (more or less... without the attention head connected to the encoder output)
    '''
    def __init__(self, num_heads, dim_embedding, dim_context, dim_feedforward, prob_dropout):
        super().__init__()
        dim_head = int(dim_embedding/num_heads)
        self.multi_head_attention = MultiHeadAttention(
            num_heads,
            dim_embedding,
            dim_head,
            dim_context,
            prob_dropout
        )
        self.feed_forward_network = FeedForwardNetwork(dim_embedding, dim_feedforward, prob_dropout)
        self.layer_norm_attention = nn.LayerNorm(dim_embedding)        
        self.layer_norm_feedfwd = nn.LayerNorm(dim_embedding)

    def forward(self, x):
        ''' forward pass
        '''
        x = x + self.multi_head_attention(
            self.layer_norm_attention(x)
        )
        x = x + self.feed_forward_network(
            self.layer_norm_feedfwd(x)
        )
        return x
