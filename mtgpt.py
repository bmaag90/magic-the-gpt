import torch
import torch.nn as nn
from torch.nn import functional
from transformer_decoder import TransformerDecoder

class mtGPT(nn.Module):
    ''' Magic the G(PT)athering Language Model

    Attributes:
        device [str]: "cuda" if available
        dim_vocuabulary [int]: dimension of vocabulary, i.e. number of possible characters
        dim_embedding [int]: dimension of embedding layer
        dim_context [int]: dimension of max. context
        dim_feedforward [int]: number of neurons in hiddenlayer of feedforward network
        num_heads [int]: number of attention heads
        num_layers [int]: number of decoder layers
        prob_dropout [flaot]: probability of dropout
    '''
    def __init__(self, dim_vocabulary, dim_embedding, dim_context, dim_feedforward, num_heads, num_layers, prob_dropout, device):
        super().__init__()

        self.device = device
        self.dim_context = dim_context
        self.input_embeddings = nn.Embedding(dim_vocabulary, dim_embedding)
        self.position_embeddings = nn.Embedding(dim_context, dim_embedding)
        self.transformer_layers = nn.Sequential(*[TransformerDecoder(num_heads, dim_embedding, dim_context, dim_feedforward, prob_dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(dim_embedding) 
        self.model_output = nn.Linear(dim_embedding, dim_vocabulary)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        ''' initialize weights
        '''
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        ''' forward pass
        '''
        batch_size, token_size = idx.shape

        # idx and targets are both (B,T) tensor of integers
        input_embeddings = self.input_embeddings(idx) # (B,T,C)
        positional_embeddings = self.position_embeddings(torch.arange(token_size, device=self.device)) # (T,C)
        x = input_embeddings + positional_embeddings # (B,T,C)
        x = self.transformer_layers(x) # (B,T,C)
        x = self.layer_norm(x) # (B,T,C)
        logits = self.model_output(x) # (B,T,dim_vocabulary)

        if targets is None:
            loss = None
        else:
            batch_size, token_size, context_size = logits.shape
            logits = logits.view(batch_size*token_size, context_size)
            targets = targets.view(batch_size*token_size)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        ''' generate tokens based on context
        '''
        for _ in range(max_new_tokens):
            # get context used for predicting next token
            idx_context = idx[:, -self.dim_context:]
            # predictions
            logits, loss = self(idx_context)
            # last token
            logits = logits[:, -1, :] 
            # probabilities
            probs = functional.softmax(logits, dim=-1) 
            # sample 
            idx_next = torch.multinomial(probs, num_samples=1) 
            # add newly sampled token
            idx = torch.cat((idx, idx_next), dim=1)         
        return idx
    

