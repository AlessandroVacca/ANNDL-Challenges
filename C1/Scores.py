import numpy as np
# ACCURACY = True Positive + True Negative / True Positive + True Negative + False Positive + False Negative
# Informally, accuracy is the fraction of predictions our model got right.
#
# PRECISION = True Positive / True Positive + False Positive
# What proportion of positive identifications was actually correct?
#
# RECALL = True Positive / True Positive + False Negative
# What proportion of actual positives was identified correctly?
#
# F1 = 2 * (Precision * Recall) / (Precision + Recall)

def get_scores(y_test, y):
    from sklearn.metrics import accuracy_score
    a = accuracy_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y, axis=1))

    from sklearn.metrics import precision_score
    p = precision_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y, axis=1))

    from sklearn.metrics import recall_score
    r = recall_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y, axis=1))

    from sklearn.metrics import f1_score
    f1 = f1_score(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y, axis=1))

    print("Accuracy: ", a)
    print("Precision: ", p)
    print("Recall: ", r)
    print("F1: ", f1)
    return a, p, r, f1
