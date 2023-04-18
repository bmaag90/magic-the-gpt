import torch
import torch.nn as nn

from attention import Attention

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention
    '''
    def __init__(self, num_heads, dim_embedding, dim_head, dim_context, prob_dropout):
        super().__init__()

        self.attention_heads = nn.ModuleList(
            [Attention(dim_embedding, dim_head, dim_context, prob_dropout) for _ in range(num_heads)]
        )
        
        self.linear_projections = nn.Linear(dim_embedding, dim_embedding)
        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, x):
        ''' forward pass
        '''
        # 3.3.2 in paper
        stacked_attention_heads = torch.cat(
            [head(x) for head in self.attention_heads],
            dim=-1
        )
        stacked_attention_heads = self.dropout(
            self.linear_projections(stacked_attention_heads)
        )
        return stacked_attention_heads