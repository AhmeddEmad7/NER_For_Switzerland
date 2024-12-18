import XlmRModel
import prepare_data
import torch
from transformers import AutoTokenizer, AutoConfig, DataCollatorForTokenClassification
from transformers.trainer import TrainingArguments, Trainer
import numpy as np
import evals
from collections import defaultdict, Counter
from datasets import concatenate_datasets
from datasets import DatasetDict
import pandas as pd
import argparse


panx_ch = prepare_data.get_data()
tags = prepare_data.tags
tags = panx_ch["de"]["train"].features["ner_tags"].feature

panx_de = panx_ch["de"].map(prepare_data.create_tag_names)

index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

xlmr_config = AutoConfig.from_pretrained("xlm-roberta-base", num_labels=tags.num_classes, id2label=index2tag, label2id=tag2index)
device = "cuda" if torch.cuda.is_available() else "cpu"
xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)

def model_init():
    return (XlmRModel.XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", config=xlmr_config).to(device))

def evaluate_lang_performance(lang, trainer):
    panx_ds = prepare_data.encode_panx_dataset(panx_ch[lang])
    return evals.get_f1_score(trainer, panx_ds["test"]), 

def concatenate_splits(corpora):
    multi_corpus = DatasetDict()
    for split in corpora[0].keys():
        multi_corpus[split] = concatenate_datasets([corpus[split] for corpus in corpora]).shuffle(seed=42)
    return multi_corpus

def align_predictions(predictions, label_ids):
    """
    Aligns model predictions with their corresponding labels, excluding ignored indices.

    This function processes batched model predictions and label IDs, converting them 
    into human-readable tag names while skipping indices marked with `-100` (ignored labels). 
    It ensures that predictions and labels are aligned at the token level.

    Args:
        predictions (numpy.ndarray): Array of shape `(batch_size, seq_len, num_labels)` 
                                     containing the model's logits for each token.
        label_ids (numpy.ndarray): Array of shape `(batch_size, seq_len)` containing 
                                   the true label IDs for each token, with `-100` 
                                   indicating ignored tokens.

    Returns:
        tuple: A pair of lists:
            - preds_list (list of list of str): Predicted tags for each example in the batch.
            - labels_list (list of list of str): True tags for each example in the batch.
    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []
    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])
                
        labels_list.append(example_labels)
        preds_list.append(example_preds)
        
    return preds_list, labels_list

def main(multi_flag=True, n_epoch=3, batch_size=24,):

##########################################Zero Shot langugage Transfer####################################################
    panx_de_encoded = prepare_data.encode_panx_dataset(panx_ch["de"])
    num_epochs = 3
    batch_size = 24
    logging_steps = len(panx_de_encoded["train"]) // batch_size
    model_name = f"xlm-roberta-base-finetuned-panx-de"

    training_args = TrainingArguments(
        output_dir=model_name, log_level="error", num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, eval_strategy="epoch",
        save_steps=1e6, weight_decay=0.01, disable_tqdm=False,
        logging_steps=logging_steps, push_to_hub=False)
    
    trainer = Trainer(model_init=model_init, args=training_args,
        data_collator=data_collator, compute_metrics=evals.compute_metrics,
        train_dataset=panx_de_encoded["train"],
        eval_dataset=panx_de_encoded["validation"],
        tokenizer=xlmr_tokenizer)
        
    trainer.train() 
    f1_scores = defaultdict(dict)
    f1_scores["de"]["de"] = evals.get_f1_score(trainer, panx_de_encoded["test"])
    print(f"F1-score of [de] model on [de] dataset: {f1_scores['de']['de']:.3f}")
    f1_scores["de"]["fr"] = evaluate_lang_performance("fr", trainer)
    print(f"F1-score of [de] model on [fr] dataset: {f1_scores['de']['fr']:.3f}")

###################################################################Mutli Linguagl##############################################
    if multi_flag:
        panx_fr_encoded = prepare_data.encode_panx_dataset(panx_ch["fr"])
        panx_de_fr_encoded = concatenate_splits([panx_de_encoded, panx_fr_encoded])
        training_args.logging_steps = len(panx_de_fr_encoded["train"]) // batch_size
        training_args.push_to_hub = False
        training_args.output_dir = "xlm-roberta-base-finetuned-panx-de-fr"
        trainer = Trainer(model_init=model_init, args=training_args,
        data_collator=data_collator, compute_metrics=evals.compute_metrics,
        tokenizer=xlmr_tokenizer, train_dataset=panx_de_fr_encoded["train"],
        eval_dataset=panx_de_fr_encoded["validation"])
        trainer.train()

        langs = ["de", "fr", "it", "en"]
        for lang in langs:
            f1 = evaluate_lang_performance(lang, trainer)
            print(f"F1-score of [de-fr] model on [{lang}] dataset: {f1:.3f}")
        
        corpora = [panx_de_encoded]
        for lang in langs[1:]:
            ds_encoded = prepare_data.encode_panx_dataset(panx_ch[lang])
            corpora.append(ds_encoded)

        corpora_encoded = concatenate_splits(corpora)
        training_args.logging_steps = len(corpora_encoded["train"]) // batch_size
        training_args.output_dir = "xlm-roberta-base-finetuned-panx-all"
        trainer = Trainer(model_init=model_init, args=training_args,
        data_collator=data_collator, compute_metrics=evals.compute_metrics,
        tokenizer=xlmr_tokenizer, train_dataset=corpora_encoded["train"],
        eval_dataset=corpora_encoded["validation"])
        trainer.train()

        for idx, lang in enumerate(langs):
            f1_scores["all"][lang] = evals.get_f1_score(trainer, corpora[idx]["test"])
            scores_data = {"de": f1_scores["de"],
            "all": f1_scores["all"]}
            f1_scores_df = pd.DataFrame(scores_data).T.round(4)
            f1_scores_df.rename_axis(index="Fine-tune on", columns="Evaluated on",
            inplace=True)
            f1_scores_df


def get_args():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--multiflag', type=bool, help="mutlilingual", required=True)
    parser.add_argument('--batchsz', type=int, help="batch size", required=False)
    parser.add_argument('--epochs',type=int, help="num epochs", required=False)

    args = parser.parse_args()
    main(args.mutliflag, args.epochs, args.batchsz)


if __name__ == "__main__":
    get_args()
