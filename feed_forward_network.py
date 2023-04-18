import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    ''' Simple Feed Forward Network
    '''
    # 3.3 in paper
    def __init__(self, dim_embedding, dim_feedforward, prob_dropout):
        super().__init__()
        # Equation 2 in Attention is all you need paper
        self.ffwd_net = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding),
            nn.Dropout(prob_dropout),
        )

    def forward(self, x):
        ''' forward pass
        '''
        return self.ffwd_net(x)
