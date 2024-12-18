import prepare_data
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def compute_metrics(eval_pred):
    y_pred, y_true = prepare_data.align_predictions(eval_pred.predictions, eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred), "precision":precision_score(y_true, y_pred),"recall":recall_score(y_true, y_pred)}

def get_f1_score(trainer, dataset):
 return trainer.predict(dataset).metrics["test_f1"]

def get_precision(trainer, dataset):
   return trainer.predict(dataset).metrics["precision"]

def get_recall(trainer, dataset):
   return trainer.predict(dataset).metrics["recall"]