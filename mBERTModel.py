from transformers import BertConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
import torch.nn as nn

class BERTForTokenClassification(BertPreTrainedModel):
    """
    A token classification model based on BERT.

    This model is designed for token-level tasks such as Named Entity Recognition (NER). 
    It uses BERT as the backbone and adds a classification head on top for predicting 
    token labels.

    Attributes:
        config_class: The configuration class for XLM-Roberta.
        num_labels (int): Number of labels for the classification task.
        bert (BertModel): The Bert model without the pooling layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Linear): A linear layer for mapping hidden states to label logits.

    Args:
        config (BertConfig): Configuration object containing the model's parameters.

    Methods:
        forward(input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
            Forward pass of the model.

            Args:
                input_ids (torch.Tensor): Tensor of input token IDs of shape `(batch_size, sequence_length)`.
                attention_mask (torch.Tensor, optional): Mask tensor of shape `(batch_size, sequence_length)` 
                                                         indicating which tokens to attend to.
                token_type_ids (torch.Tensor, optional): Tensor of shape `(batch_size, sequence_length)` 
                                                        specifying token types (not typically used in BERT models).
                labels (torch.Tensor, optional): Tensor of shape `(batch_size, sequence_length)` containing 
                                                 the true labels for each token.

            Returns:
                TokenClassifierOutput: An output object containing:
                    - `loss` (torch.Tensor, optional): The computed loss if `labels` are provided.
                    - `logits` (torch.Tensor): Logits of shape `(batch_size, sequence_length, num_labels)`.
                    - `hidden_states` (tuple, optional): Hidden states from the backbone model.
                    - `attentions` (tuple, optional): Attention weights from the backbone model.
    """
    
    config_class = BertConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, **kwargs):
        # Use model body to get encoder representations
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, **kwargs)
        
        # Apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        
        # Calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        # Return model output object
        return TokenClassifierOutput(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions
        )