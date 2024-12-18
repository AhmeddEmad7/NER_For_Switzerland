import XlmRModel
import prepare_data
import torch
from transformers import AutoTokenizer, AutoConfig


panx_ch = prepare_data.get_data()
tags = prepare_data.tags
tags = panx_ch["de"]["train"].features["ner_tags"].feature

panx_de = panx_ch["de"].map(prepare_data.create_tag_names)

index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

xlmr_config = AutoConfig.from_pretrained("xlm-roberta-base", num_labels=tags.num_classes, id2label=index2tag, label2id=tag2index)
device = "cuda" if torch.cuda.is_available() else "cpu"

xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


