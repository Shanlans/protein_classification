import numpy as np
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt

def convert_multi_label_for_cm(y_pred,y_actual,class_num,prob_threshold=0.5):
    PRED_COUNT = y_actual.size(0)
    temp_array = np.arange(0,class_num).reshape((1,class_num))
    temp_array = temp_array.repeat(PRED_COUNT,axis=0)
    
    pred = np.where(y_pred > prob_threshold,temp_array,0).reshape(-1)
    
    actual = y_actual.cpu().numpy()
    actual = np.where(actual>0,temp_array,28).reshape(-1)
    return pred,actual




def print_confusion_matrix(confusion_matrix, class_names,normalize,figsize = (15,15), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fmt = '.2f' if normalize else 'd'
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


    def draw_cm(pred,label,class_name,class_num,prob_threshold=0.5):
        p,l=convert_multi_label_for_cm(pred,label,class_num=class_num,prob_threshold=prob_threshold)
        cln = class_name.copy()
        cln.append("UNK")
        cnf_matrix = confusion_matrix(l, p,labels=list(range(len(cln))))
        fig = print_confusion_matrix(cnf_matrix, cln,normalize=False,figsize = (15,15), fontsize=14)
    return fig


def accuracy(y_pred, y_actual, prob_threshold=0.5):
        """ """
        final_acc = 0
        # for prob_threshold in np.arange(0, 1, 0.01):
        PRED_COUNT = y_actual.size(0)
        PRED_CORRECT_COUNT = 0
        pred = np.where(y_pred > prob_threshold, np.ones_like(y_pred,dtype=np.int32), 0)
        for j in range(pred.shape[0]):
            if torch.equal(y_actual[j],torch.tensor(pred[j]).float()):
                PRED_CORRECT_COUNT += 1
        if PRED_COUNT == 0:
            final_acc = 0
        else:
            final_acc = PRED_CORRECT_COUNT / PRED_COUNT
        return final_acc * 100, PRED_COUNT