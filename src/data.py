import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
roc_curve, precision_recall_curve,roc_auc_score,accuracy_score)
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler,SMOTE

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats



class WayfairDataset(object):
    def __init__(self,data_path):
        super(WayfairDataset, self).__init__()

        self.data_path = data_path
        self.df = pd.read_csv(data_path)




rawdata_train = rawdata_train.replace('NaN',np.NaN) # making the missing be Nan
rawdata_train.fillna(rawdata_train.mean(), inplace=True) # fill with mean



























if __name__ =='__main__':

    data_path = 'data/df_holdout_scholarjet.csv'
    rawdata_test = pd.read_csv(data_path,index_col = 0)
    rawdata_train = pd.read_csv('data/df_training_scholarjet.csv',index_col = 0)