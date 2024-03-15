import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score, auc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

def find_high_corr_pairs(df, threshold=0.8):
    corr_matrix = df.corr()  # Compute the correlation matrix
    high_corr_pairs = []  # Initialize an empty list to store the high-correlation pairs

    # Iterate over the lower triangle of the correlation matrix to avoid duplicate pairs
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # Check if absolute correlation exceeds the threshold
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    return high_corr_pairs

def manual_pr_draw(y_test, y_pred):
    # Calculate the Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    
    # Plot the Precision-Recall curve
    # plt.subplot(1,2,2)
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = %0.3f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

def manual_auc_roc_draw(y_test, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def groupper(groups_):
    prev_num = int()
    lst = []
    counter = 1
    for i, num in enumerate(groups_):
        if i != 0:
            if num == prev_num:
                counter += 1
            elif num != prev_num:
                lst.append(counter)
                counter = 1
        prev_num = num
    lst.append(counter)
    return lst

def feature_importance(col, coef, model_name = ""):
    # col = X_train_fe_used.columns
    # coef = model.feature_importances_
    coef_df = pd.DataFrame({'Feature': col, 'Coefficient': coef})
    coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

    plt.figure(figsize=(10, 6))  # You can adjust the figure size as per your requirements
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')  # Horizontal bar chart for better readability
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.title(f'Importance of Features in Predicting Result ({model_name} Features)')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
    plt.show()