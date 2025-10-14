from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, \
    accuracy_score, f1_score, precision_score, recall_score, mean_squared_error

def evaluate_metrics(y_true, y_prob, task):
    y_pred = [int(p > 0.5) for p in y_prob] 
    if task == 'BCE':
        return {
            'AUC': roc_auc_score(y_true, y_prob),
            'PR-AUC': average_precision_score(y_true, y_prob),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)
        }
    elif task == 'MSE':
        return {
            'RMSE': mean_squared_error(y_true, y_prob, squared=False),
            'Pearson': pearsonr(y_true, y_prob)[0],
            'Spearman': spearmanr(y_true, y_prob)[0]
        }
