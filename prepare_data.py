from datasets import load_dataset
from collections import defaultdict, Counter
from datasets import DatasetDict
import pandas as pd
import torch
tags=None


def get_data():
    
    langs = ["de", "fr", "it", "en"]
    panx_ch = defaultdict(DatasetDict)
    
    for lang in langs:
        ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
        for split in ds: 
            panx_ch[lang][split] = ds[split].shuffle(seed=0)
        
    return panx_ch

def create_tag_names(batch):
 return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}

def tag_text(text, tags, model, tokenizer, device="cuda"):
    """
    Tags a given text with predictions from a NER model.

    This function tokenizes the input text, runs it through a NER model, and 
    generates predictions for each token. The predictions are converted into 
    human-readable tag names, and the tokens and their corresponding tags are 
    returned as a pandas DataFrame.

    Args:
        text (str): The input text to be tagged.
        tags: A mapping or object containing tag names (e.g., `tags.names`).
        model: A pre-trained NER model that takes tokenized inputs and 
               outputs logits for each token.
        tokenizer: A tokenizer that splits the input text into tokens 
                   compatible with the model.
    
    Returns:
        pd.DataFrame: A DataFrame containing:
            - "Tokens": List of tokens from the input text.
            - "Tags": Predicted tags corresponding to each token.
    """
    tokens = tokenizer(text).tokens()
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids)[0]
    predictions = torch.argmax(outputs, dim=2)
    preds = [tags.names[p] for p in predictions[0].cpu().numpy()]
    return pd.DataFrame([tokens, preds], index=["Tokens", "Tags"])

def tokenize_and_align_labels(examples, xlmr_tokenizer):
    """
    Tokenizes input sentences and aligns NER labels with tokenized outputs.

    This function uses a tokenizer that supports word-level tokenization and aligns 
    the NER tags to the subword tokenization scheme. It assigns `-100` to subword 
    tokens or special tokens to ensure they are ignored during the loss computation.

    Args:
        examples (dict): A dictionary containing:
            - "tokens" (list of list of str): Sentences represented as lists of tokens.
            - "ner_tags" (list of list of int): Corresponding NER tags for the tokens.

    Returns:
        dict: A dictionary containing:
            - Tokenized inputs (e.g., "input_ids", "attention_mask").
            - "labels": Aligned labels for the tokenized inputs.
    """
    tokenized_inputs = xlmr_tokenizer(examples["tokens"], truncation=True, 
                                      is_split_into_words=True)
    labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def encode_panx_dataset(corpus):
    """
    Encodes a PAN-X dataset by tokenizing the input sentences and aligning NER labels.

    This function applies the `tokenize_and_align_labels` function to the dataset using 
    batched processing, removing unnecessary columns (e.g., 'tokens', 'ner_tags', 'langs') 
    to prepare the dataset for model training.

    Args:
        corpus (DatasetDict): A `DatasetDict` object containing splits (e.g., 'train', 
                              'validation', 'test') with the features:
                              - "tokens": List of tokens for each sentence.
                              - "ner_tags": NER labels for the tokens.
                              - "langs": Language identifiers.

    Returns:
        DatasetDict: A `DatasetDict` with tokenized inputs and aligned labels, 
                     containing features such as "input_ids", "attention_mask", 
                     and "labels".
    """
    return corpus.map(tokenize_and_align_labels, batched=True, 
                      remove_columns=['tokens', 'ner_tags', 'langs'])