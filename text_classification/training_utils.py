from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate

metric = evaluate.load('accuracy')
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=-1)  # Get predicted class labels
    return metric.compute(predictions=preds, references=labels)


def getClassWts(df):
    return compute_class_weight("balanced", classes=sorted(df['label'].unique().tolist(), y=df['label'].tolist()))
                         