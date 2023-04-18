# -*- coding: utf-8 -*-
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def concatenate_text(row, columns_keep):
    ''' concatenate text, i.e. the different fields/columns, of a card
    '''
    x = ''
    if 'name' in columns_keep:
        x += '{}, '.format(row['name'])
    if 'manaCost' in columns_keep:
        x += '{}, '.format(row['manaCost'])
    if 'type' in columns_keep:
        x += '{}, '.format(row['type'])
    if 'text' in columns_keep:
        x += '{}, '.format(row['text'])
    if 'power' in columns_keep:
        x += 'power {}, '.format(row['power'])
    if 'toughness' in columns_keep:
        x += 'toughness {}, '.format(row['toughness'])
    # remove last ', '
    x = x[:-2]

    return x
    
def generate_card_text(df, columns_keep, convert_to_lowercase, remove_special_chars):
    ''' generate a single line of text for each card
    '''
    # remove duplicates, i.e. multiple printings
    dat = df[columns_keep].drop_duplicates()
    # concatenate columns to text
    concat_text = dat.apply(lambda x: concatenate_text(x, columns_keep), axis=1).to_list()
    # lowercase
    if convert_to_lowercase:
        concat_text = [x.lower() for x in concat_text]
    # remove special characters
    if remove_special_chars is not None:
        concat_text = [re.sub(remove_special_chars, '', x) for x in concat_text]
    # remove newline
    concat_text = [re.sub('\n', ' ', x) for x in concat_text]

    return concat_text
    
def load_csv_file(csv_filepath):
    ''' load input csv file
    '''
    df = pd.read_csv(csv_filepath, low_memory=False)

    return df

def save_txt_file(arr_text_data, txt_filepath):
    ''' save processed text to txt file
    '''
    with open(txt_filepath, 'w') as fh:
        for x in arr_text_data:
            fh.write(x + '\n')

def pre_process_data(csv_filepath, txt_filepath, columns_keep, convert_to_lowercase, remove_special_chars, train_data_ratio=1.0):
    ''' run preprocessing
    '''
    df = load_csv_file(csv_filepath)

    concat_text = generate_card_text(
        df, 
        columns_keep, 
        convert_to_lowercase, 
        remove_special_chars
        )
    
    if train_data_ratio < 1.0:
        train_concat_text, val_concat_text = train_test_split(
            concat_text, 
            train_size=train_data_ratio
            )
        save_txt_file(
            train_concat_text,
            re.sub('.txt', '_train.txt', txt_filepath)
            )
        save_txt_file(
            val_concat_text, 
            re.sub('.txt', '_val.txt', txt_filepath)
            )
    else:
        save_txt_file(concat_text, txt_filepath)

if '__name__' == '__main__':

    pre_process_data(
        'data/AllPrintingsCSVFiles/cards.csv', 
        'data/preproc/cards_text.txt', 
        ['name','manaCost','type','text','power','toughness'], 
        True, 
        '[½®π∞☐àáâéíöúû−•²]', 
        train_data_ratio=0.9
        )