# Magic the GPT
Magic The GPT - GPT inspired model to generate Magic the Gathering cards

## Description

This project tries to answer the question if a Transformer decoder model is able to generate better Magic the Gathering cards than Wizards of the Cost.
The model follows the description in the "Attention is all you need" paper (Vaswani et al. 2017) and is trained on MTG cards, which can be downloaded from [https://mtgjson.com/](https://mtgjson.com/).
The cards are first pre-processed, i.e. relevant information concatentated to convert the csv file into text, and then used to train the model, which predicts the next character based on some context.

Finally, the model is able to generate cards like a Spirit featuring Cascade:
```
spirit supprener, {3}{g}, creature — spirit, 
reach (this creature can block creatures with flying.) 
cascade (when you cast this spell, exile cards from the top of your library until you exile a nonland card that costs less. 
you may cast it without paying its mana cost. put the exiled cards on the bottom of your library in a random order.), 
power 6, toughness 5
```

Some interesting designs (not sure how one would sacrifice Dark Ritual to be honest...):
```
demigolish, {3}{b}, creature  insect, 
sacrifice dark ritual: it deals 2 damage to target creature., 
power 3, toughness 1
```

Or a completely useless counterspell:
```
rot of the wild, {4}{g}{g}{u}, sorcery, 
counter target spell., 
power nan, toughness nan
```

A best-of with some more interesting cards can be found in `results/bestof.txt` and additional examples in `generated_cards.txt` in the individual models directories.

# How-to-use

## Implementation
The notebook `train_mtGPT_model.ipynb` shows how to pre-process the data, train a model and generate new cards.
The implementation of the different is found as follows:
* Data preprecossing: 
    * `data_preprocessing.py`: converts the csv file into a txt file
    * `mtg_dataset`: pytorch Dataset used for pytorch training and validation DataLoader
* Model:
    * `attention.py`: Basic attention mechanism, see Section 3.2.1 in paper
    * `feed_forward_network.py`: Simple feedforward network with 2 linear layers
    * `multi_head_attention.py`: Multi-Head attention, i.e. concatenation of multiple attention heads, see Section 3.2.2
    * `transformer_decoder.py`: Decoder stack, which combines the multi-head attention and feed-forward networks
    * `mtgpt.py`: Implements main model class, which combines/stacks multiple transformer-decoders to the final model
* Utility functions:
    * `utility.py`: Different utility functions including encoding and decoding of text and tokens resp., loss estimation and saving the trained model

## Configuration
The specific model and training settings can be set in the `config.yaml` file:
* `train`:
    * `batch_size`: Batch size 
    * `epochs`: Number of training epochs
    * `eval_interval`: Interval of epochs when loss will be estimated and printed
    * `learning_rate`: Learning rate of optimizer
    * `eval_iters`: Iterations/num. batches performed to estimate loss
    * `max_batches_per_epoch`: Max. number of batches used per epoch (training on all data would take a considerable amount of time)
* `model`:
    * `dim_context`: Number of tokens/characters that are used as context for predictions
    * `dim_embedding`: Dimensions of the embedding layer
    * `dim_feedforward`: Dimensions of the feed-foward linear layer, should be roughly 4x the dimension of the embedding layer according to the paper
    * `num_heads`: Number of attention heads
    * `num_layers`: Number of decoder layers that will be stacked
    * `prob_dropout`: Dropout probability

# Trained models
The models are saved to the timestamped folders in the `models` directory. The following trained models are already available:
1. `models/2304161750`: 10 Mio. parameter model
2. `models/2304200902`: Same 10 Mio. parameter model, but trained on additional 10 epochs. Shows signs of overfitting, i.e. validation loss increased
3. `models/2304221751`: 1.3 Mio parameters model, higher loss than 10 Mio. parameter models and, thus, also generates cards of lower quality
