import os
import numpy as np
from keras.models import load_model
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import linear_model
import copy
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from itertools import cycle

def get_accuracy(all_gt, all_label):
    return len(np.argwhere(all_gt==all_label))/float(len(all_gt))

def Youden_Cutoff_auc_sen_spc(y_true, y_prob):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    y_true : Matrix with dependent or target data, where rows are observations

    y_pred : Matrix with predicted data, where rows are observations

    Returns
    -------     
    auc, youden sensitivity and youden specificity
    """

    fpr, tpr, threshold = roc_curve(y_true, y_prob)

    spc = 1-fpr;
    j_scores = tpr-fpr
    best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]

    predicted_label = copy.deepcopy(y_prob)
    predicted_label[predicted_label>youden_thresh] = 1
    predicted_label[predicted_label<youden_thresh] = 0
    accuracy = get_accuracy(y_true, predicted_label)
    print(accuracy)

    auc = roc_auc_score(y_true, y_prob)

    return auc, youden_sen, youden_spc, fpr, tpr, threshold

def auc_95confidence(y_true,y_prob):

    # https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals

    n_bootstraps = 1000
    bootstrapped_scores = []
    np.random.seed(seed)

    for i in range(n_bootstraps):
        indices = np.random.choice(range(0, len(y_prob)), len(y_prob), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower, confidence_upper

def compute_metrics(y_true, y_prob):
    """
    y_true: true label
    y_prob: probability score for class 1
    """
    y_pred = np.round(y_prob)

    acc = get_accuracy(y_true, y_pred)
    auc, youden_sen, youden_spc, fpr, tpr, threshold = Youden_Cutoff_auc_sen_spc(y_true, y_prob)
    lower, upper = auc_95confidence(y_true, y_prob)

    return acc, youden_sen, youden_spc, auc, lower, upper, fpr, tpr, threshold

def organize_result(result_dict, case, y_true, y_prob):

    result_dict[case] = {}
    
    acc, sen, spe, youden_auc, lower, upper, fpr, tpr, threshold = compute_metrics(y_true, y_prob)

    result_dict[case]['Accuracy'] = acc
    result_dict[case]['Sensitivity'] = sen
    result_dict[case]['Specificity'] = spe
    result_dict[case]['AUC'] = youden_auc
    result_dict[case]['AUC Upper'] = upper
    result_dict[case]['AUC Lower'] = lower
    result_dict[case]['fpr'] = fpr
    result_dict[case]['tpr'] = tpr
    result_dict[case]['threshold'] = threshold

    return result_dict

def draw_ROC_curve(result_dict):

    fig_handle=plt.figure()
    colors = ['deeppink', 'darkorange','navy','aqua', 'cornflowerblue','lime','orchid']
    lw = 2

    # organize in order of smallest to largest AUC
    ordered_list = list()

    for case in result_dict.keys():
        ordered_list.append((case,result_dict[case]['AUC'],result_dict[case]['fpr'],result_dict[case]['tpr']))

    ordered_list = sorted(ordered_list, key = lambda case: case[1])

    #plot
    for ind in range(len(ordered_list)):
        # print(ordered_list[ind][0])

        # plt.plot(ordered_list[ind][2], ordered_list[ind][3], color = colors[ind], lw= lw,
        #         label = '{0} ROC curve (area = {1:0.3f})'
        #         ''.format(ordered_list[ind][0], ordered_list[ind][1]))
        plt.plot(ordered_list[ind][2], ordered_list[ind][3], color = colors[ind], lw= lw)

    plt.plot([0,1],[0,1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.setp(ax.get_xticklabel(), visible=False, ax.get_yticklabel(), visible=False)
    # plt.tick_params(axis='both', which= 'major', labelsize = 18)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiveer Operating Characteritic for test set')
    # plt.legend(loc="lower right")

    plt.show()

    return None

#Specify which GPU to be used
os.environ["CUDA_VISIBLE_DEVICES"]="0"
seed=0

# load test data
os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced_20/test')
test_FLAIR = np.load('test_FLAIR.npy')
test_T2 = np.load('test_T2.npy')
test_T1 = np.load('test_T1.npy')
test_T1post = np.load('test_T1post.npy')

test_1p19q = np.load('test_1p19q.npy')
test_1p19q = test_1p19q.astype(np.float32).reshape(-1,1)
test_age = np.expand_dims(np.load('test_age.npy'),1)
test_KPS = np.expand_dims(np.load('test_KPS.npy'),1)
test_gender = np.expand_dims(np.load('test_gender.npy'),1)

#load saved CNN models
os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced_20/outputs/models/')
model_FLAIR = load_model('flair_model_conv2.h5')
model_T2 = load_model('T2_model_conv2.h5')
model_T1 = load_model('T1_model_conv2.h5')
model_T1post = load_model('T1post_model_conv2.h5')

#load saved logreg model parameters
os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced_20/outputs/logreg/')

logreg_imaging_param = np.load('logreg_imaging.npy').item()
logreg_imaging_age_param = np.load('logreg_imaging_age.npy').item()
logreg_all_param = np.load('logreg_all.npy').item()

# Call logistic regression from scikit-learn and set the parameters
logreg_imaging = LogisticRegression()
for key,value in logreg_imaging_param.items():
    setattr(logreg_imaging, key, value)

logreg_imaging_age = LogisticRegression()
for key,value in logreg_imaging_age_param.items():
    setattr(logreg_imaging_age, key, value)

logreg_all = LogisticRegression()
for key,value in logreg_all_param.items():
    setattr(logreg_all, key, value)

# obtain sigmoid probs for test set
test_sig_FLAIR = model_FLAIR.predict(test_FLAIR,batch_size=16)
test_sig_T2 = model_T2.predict(test_T2,batch_size=16)
test_sig_T1 = model_T1.predict(test_T1,batch_size=16)
test_sig_T1post = model_T1post.predict(test_T1post,batch_size=16)

# pre-define a dictionary to save results
result = {}

# Case1: Consider only 1 modality sequence: FLAIR, T2, T1, T1post
organize_result(result, 'FLAIR', test_1p19q, test_sig_FLAIR)
organize_result(result, 'T2', test_1p19q, test_sig_T2)
organize_result(result, 'T1', test_1p19q, test_sig_T1)
organize_result(result, 'T1post', test_1p19q, test_sig_T1post)

# Case2: Consider all imaging modality combined with logistic regression as a classifier
test_imaging = np.hstack((test_sig_FLAIR,test_sig_T2, test_sig_T1, test_sig_T1post))

Z_imaging = logreg_imaging.predict_proba(test_imaging)[:,1].reshape(-1,1)
organize_result(result, 'Imaging combined', test_1p19q, Z_imaging)

# Case3: imaging modality + age with logreg
test_imaging_age = np.hstack((test_sig_FLAIR,test_sig_T2, test_sig_T1, test_sig_T1post, test_age))

Z_imaging_age = logreg_imaging_age.predict_proba(test_imaging_age)[:,1].reshape(-1,1)
organize_result(result, 'Imaging combined + age', test_1p19q, Z_imaging_age)

# Case4: imaging modality + age + KPS + gender with logreg
test_all = np.hstack((test_sig_FLAIR,test_sig_T2, test_sig_T1, test_sig_T1post, test_age, test_KPS, test_gender))

Z_all = logreg_all.predict_proba(test_all)[:,1].reshape(-1,1)
organize_result(result, 'Imaging combined + age + KPS + gender', test_1p19q, Z_all)

# plot ROC curve
draw_ROC_curve(result)
plt.savefig('ROC.tif', dpi=300)

#plot ROC with combined + age
result_copy = copy.deepcopy(result)
del result_copy['Imaging combined + age']

draw_ROC_curve(result_copy)

df = pd.DataFrame.from_dict(result)

os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced_20/outputs/')
df.to_csv('result_summary.csv')