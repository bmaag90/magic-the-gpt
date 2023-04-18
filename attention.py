import torch
import torch.nn as nn
from torch.nn import functional


class Attention(nn.Module):
    '''
    '''

    def __init__(self, dim_embedding, dim_head, dim_context, prob_dropout):
        super().__init__()

        self.key = nn.Linear(
            dim_embedding,
            dim_head,
            bias=False
        )

        self.query = nn.Linear(
            dim_embedding,
            dim_head,
            bias=False
        )

        self.value = nn.Linear(
            dim_embedding,
            dim_head,
            bias=False
        )

        self.register_buffer(
            'tril',
            torch.tril(torch.ones(dim_context, dim_context))
        )

        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, x):

        batch_size, d_token, dim_context = x.shape

        key = self.key(x)
        query = self.query(x)
        
        # Equation 1 in original paper
        attention = query @ key.transpose(-2,-1)*(key.shape[-1]**-0.5)
        masked_attention = attention.masked_fill(
            self.tril[:d_token, :d_token] == 0, float('-inf')
        )
        masked_attention = functional.softmax(masked_attention, dim=-1)
        masked_attention = self.dropout(masked_attention)
        
        value = self.value(x)
        y = masked_attention @ value

        return y
