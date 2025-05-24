import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
# import seaborn as sns
import pdb, traceback
# from utils.lib import blockPrintingIfServer


def build_ROC(plt, gold, pred, log_log = False):
    try:
        fpr, tpr, thresholds = roc_curve(gold, pred)
        roc_auc = roc_auc_score(gold, pred)
        if log_log:
            # Plot the ROC curve in log-log scale    
            plt.loglog(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.loglog([1e-6, 1], [1e-6, 1], color='gray', linestyle='--')
            plt.xlim([1e-6, 1])
            plt.ylim([1e-6, 1])
            plt.xlabel('False Positive Rate (log)')
            plt.ylabel('True Positive Rate (log)')
            plt.title('ROC Curve (log-log scale)')
        else:
            # Plot the ROC curve            
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()


    



# @blockPrintingIfServer
def plot_ROC_curve(gold, pred, log_log = False):
    plt.figure(figsize=(8, 6))
    build_ROC(plt, gold, pred, log_log)   

    plt.legend(loc='lower right')
    plt.show()

    plot_CM(gold, pred)

# @blockPrintingIfServer
# def plot_CM(gold, pred):
#     # Compute confusion matrix
#     cm = confusion_matrix(gold, pred)

#     # Create a heatmap of the confusion matrix
#     class_names = ['Negative', 'Positive']  # Replace with your class names
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#                 xticklabels=class_names, yticklabels=class_names) # type: ignore
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.show()