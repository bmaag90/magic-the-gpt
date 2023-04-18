from torch.utils.data import Dataset
import pandas as pd
import re

class MTGDataset(Dataset):

    def __init__(self, card_text_data, dim_context):
        ''' MTG Dataset
        '''
        self.card_text_data = card_text_data
        self.dim_context = dim_context

    def __len__(self):
        ''' Length of dataset equals length of text minus the length of each sample, i.e. dim_context
        '''
        return len(self.card_text_data) - self.dim_context
    
    def __getitem__(self, idx):
        ''' x is a string of length dim_context starting at index idx
        y is the same string shifted by 1 character 
        i.e. we predict the next character given by y based on contex x
        '''
        x = self.card_text_data[idx:idx+self.dim_context]
        y = self.card_text_data[idx+1:idx+self.dim_context+1]
        return x, y