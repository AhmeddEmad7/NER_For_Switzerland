import XlmRModel
import prepare_data
import torch
from transformers import AutoTokenizer, AutoConfig, DataCollatorForTokenClassification
from transformers.trainer import TrainingArguments, Trainer
import numpy as np
import evals

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

def main(zero_shot_flag=True, multi_flag=True, n_epoch=3, batch_size=24,):

    if zero_shot_flag:
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
        data_collator=data_collator, compute_metrics=compute_metrics,
        train_dataset=panx_de_encoded["train"],
        eval_dataset=panx_de_encoded["validation"],
        tokenizer=xlmr_tokenizer)
        


