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
    def __init__(self,train,test,drop_missing_ratio):
        super(WayfairDataset, self).__init__()

        self.drop_missing_ratio = drop_missing_ratio
        self.features = features
        self.y_train = y_train


    def drop_missing(self):
        pass
        

    def finalfill(self):
        objects = []
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerics = []
        for i in self.features.columns:
            if self.features[i].dtype == object:
                objects.append(i)
            if self.features[i].dtype in numeric_dtypes:
                numerics.append(i)
        self.features.update(self.features[objects].fillna('None'))
        self.features.update(self.features[numerics].fillna(0))

    def mainClean(self):
        print('Cleaning ~')
        self.toCat()
        self.defaultfill()
        self.zerofill()
        self.modefill()
        self.catnumfiil()
        self.nonefill()
        self.finalfill()
        print('Cleaned !!! ')

        return self.features, self.y_train


























if __name__ =='__main__':

    data_path = 'data/df_holdout_scholarjet.csv'
    rawdata_test = pd.read_csv(data_path,index_col = 0)
    rawdata_train = pd.read_csv('data/df_training_scholarjet.csv',index_col = 0)