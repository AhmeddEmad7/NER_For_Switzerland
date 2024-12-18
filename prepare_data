from datasets import get_dataset_config_names
from datasets import load_dataset
from collections import defaultdict, Counter
from datasets import DatasetDict
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from transformers import XLMRobertaConfig, AutoConfig, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.nn.functional import cross_entropy
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


tags=None


def get_data():
    
    langs = ["de", "fr", "it", "en"]
    panx_ch = defaultdict(DatasetDict)
    
    for lang in langs:
        ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
        for split in ds: 
            panx_ch[lang][split] = ds[split].shuffle(seed=0)
        
    return panx_ch

#panx_ch = get_data()

def create_tag_names(batch):
 return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}

#panx_de = panx_ch["de"].map(create_tag_names)

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