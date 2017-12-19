import requests
import numpy as np
import scipy as sp
import sys
import platform
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.cross_validation import StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier ,ExtraTreesClassifier,AdaBoostClassifier, BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import *
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import re
import random
import heapq
import warnings
from math import sqrt, exp, log
from csv import DictReader
from sklearn.preprocessing import Imputer
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV , RandomizedSearchCV, ParameterSampler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint
from sklearn import decomposition, pipeline, metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score,roc_curve,auc
import collections
import ast
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression, \
    Perceptron,RidgeCV, TheilSenRegressor
from datetime import date,timedelta as td,datetime as dt
import datetime
from sklearn.feature_selection import SelectKBest,SelectPercentile, f_classif, GenericUnivariateSelect
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder

########################################################################################################################
#ChatBot test
########################################################################################################################
########################################################################################################################
class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values,data.Utterance.values))

    def predict(self, context, utterances):

        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)

        print(np.shape(vector_context))
        print(np.shape(vector_doc))

        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]

########################################################################################################################
def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

########################################################################################################################
# Random Predictor
def predict_random(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)

########################################################################################################################
# Evaluate Random predictor
def Random_predictor(test_df):
    y_random = [predict_random(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
    y_test = np.zeros(len(y_random))
    for n in [1, 2, 5, 10]:
        print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y_random, y_test, n)))

    return test_df

########################################################################################################################
#Cross Validation and model fitting
########################################################################################################################
def Nfold_Cross_Valid(X, y, clf):

    print("***************Starting Kfold Cross validation***************")

    X =np.array(X)
    scores=[]

    # lbl = preprocessing.LabelEncoder()
    # lbl.fit(list(y))
    # y = lbl.transform(y)

    ss = StratifiedShuffleSplit(y, n_iter=5,test_size=0.2)
    #ss = KFold(len(y), n_folds=5,shuffle=False,indices=None)

    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        y_train, y_test= y[trainCV], y[testCV]

        #clf.fit(X_train, y_train, early_stopping_rounds=500, eval_metric=customized_eval,eval_set=[(X_test, y_test)])
        clf.fit(X_train, y_train)

        y_pred=clf.predict_proba(X_test)
        # temp = pd.DataFrame(y_pred)
        y_pred = get_best_five(y_pred,type_val=True)

        scores.append(score_predictions(y_pred, pd.DataFrame(y_test)).mean())

        # temp = pd.concat([temp,y_pred],axis=1)
        # temp['test'] = y_test
        # temp['val'] = score_predictions(y_pred, pd.DataFrame(y_test))
        # temp.to_csv(file_path+'temp.csv')

        print(" %d-iteration... %s " % (i,scores))

        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    print("***************Ending Kfold Cross validation***************")

    return scores

########################################################################################################################
#Data cleansing , feature scalinng , splitting
########################################################################################################################
def Data_Munging(Train_DS,Actual_DS, Val_DS):

    print("***************Starting Data cleansing***************")

    global  Train_DS1,Actual_DS1, lbl_y

    #test_df = Random_predictor(Actual_DS)

    # Evaluate TFIDF predictor
    pred = TFIDFPredictor()
    pred.train(Train_DS)

    y = [pred.predict(Actual_DS.Context[x], Actual_DS.iloc[x, 1:].values) for x in range(len(Actual_DS))]

    y_random = [predict_random(Actual_DS.Context[x], Actual_DS.iloc[x, 1:].values) for x in range(len(Actual_DS))]
    y_test = np.zeros(len(y_random))

    for n in [1, 2, 5, 10]:
        print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y, y_test, n)))

    sys.exit(0)

    # pd.DataFrame(Train_DS).to_csv(file_path+'Train_DS_50000.csv')
    print("***************Ending Data cleansing***************")

    return Train_DS, Actual_DS, y

########################################################################################################################
#XGB_Classifier
########################################################################################################################
def XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid):

    print("***************Starting XGB Classifier***************")
    t0 = time()

    if Grid:
       #used for checking the best performance for the model using hyper parameters
        print("Starting model fit with Grid Search")

        param_grid = {'n_estimators': [50],
                      'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,20],
                      'min_child_weight': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20],
                      'subsample': [0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9,1],
                      'colsample_bytree': [0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9,1],
                      'silent':[True],
                      'gamma':[2,1,0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9]
                     }

        #run randomized search
        n_iter_search = 800
        clf = xgb.XGBClassifier(nthread=8)
        clf = RandomizedSearchCV(clf, param_distributions=param_grid,
                                           n_iter=n_iter_search, scoring = 'log_loss',cv=3)
        start = time()
        clf.fit(np.array(Train_DS), np.array(y))

        print("GridSearchCV completed")
        Parms_DS_Out = report(clf.grid_scores_,n_top=n_iter_search)
        Parms_DS_Out.to_csv(file_path+'Parms_DS_XGB_4.csv')

        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        sys.exit(0)
    else:
        ##----------------------------------------------------------------------------------------------------------------##
        #best lb is with n_estimators = 500 , using 1000 it is less
        #CV: 0.78526434774405007 (full set)
        #CV: 0.824999 (100k set - with Age set up, all dummy)
        #CV: 0.830194 (with 50 K) - n_estimators = 75 - with Age Bkt and Session (Action_Type dummy) features
        #CV: 0.830842 (with 50 K) - n_estimators = 75 - with Age Bkt and Session & Session 3 - features  *********

        clf = xgb.XGBClassifier(n_estimators=500,nthread=8)

        #LB : n_estimators = 100 , 0.88040
        #LB : n_estimators = 125 , 0.88059 ***best, session 1,2,3,4
        #LB : n_estimators = 150 , 0.88045
        #LB : n_estimators = 060 , 0.87996
        #LB : n_estimators = 080 , 0.88029
        #LB:  n_estimators = 125 , 0.88148 ***best, session 1,2,3,4 and year > 2012
        #LB:  n_estimators = 125 , 0.88080, session 1,2,3,4 and year > 2013
        #LB:  n_estimators = 125 , 0.88010, session 1,2,3,4 and year > 2011

        #CV: 0.83062 (with 50 K) - n_estimators = 125 - with Age Bkt and Session,2,3,4 - features  *********


        clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
                                gamma=0.6, learning_rate=0.1, max_delta_step=0, max_depth=6,
                                min_child_weight=12, missing=None, n_estimators=135, nthread=8,
                                objective='multi:softprob', reg_alpha=0, reg_lambda=1,
                                scale_pos_weight=1, silent=True, subsample=0.7)

        #clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid')

        # clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.25, n_estimators=75,
        #             objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0,nthread=8)

        ##-----------------------------------
        # Nfold_score = Nfold_Cross_Valid(Train_DS, y, clf)
        # sys.exit(0)

        X_train = np.array(Train_DS)
        Y_train = np.array(y)

        clf.fit(X_train, Y_train)

    X_Actual = np.array(Actual_DS)

    #Predict actual model
    pred_Actual = clf.predict_proba(X_Actual)
    print("Actual Model predicted")

    if raw_output == False:
        pred_Actual = get_best_five(pred_Actual,type_val=False)

        #Get the predictions for actual data set
        pred_Actual.to_csv(file_path+'output/Submission_Roshan_xgb_135.csv', index_label='id')

    else:

        print(pd.DataFrame(pred_Actual).head())
        pred = pd.DataFrame(pred_Actual)
        pred['id'] = Actual_DS1
        pred = pred.set_index('id')


        pred.to_csv(file_path+'output/Submission_Roshan_xgb_raw_150_2012.csv', index_label='id')


    print("***************Ending XGB Classifier***************")
    return pred_Actual

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, Train_DS1, Session_DS, Age_DS, Session_DS3, Session_DS2,Session_DS4,Session_DS5,Session_DS6, raw_output, Lang_DS

    # Mlogloss
    #Mlogloss_scorer = metrics.make_scorer(multiclass_log_loss, greater_is_better = False)

    raw_output = False
    random.seed(42)
    np.random.seed(42)

    if(platform.system() == "Windows"):

        file_path = 'C:\\Python\\Tes\\ChatBot\\data_folder\\data\\'
    else:
        file_path = ''

########################################################################################################################
#Read the input file , munging and splitting the data to train and test
########################################################################################################################
    Train_DS      = pd.read_csv(file_path+'train.csv',sep=',').head(1000)
    Actual_DS     = pd.read_csv(file_path+'test.csv',sep=',').head(1000)
    Val_DS        = pd.read_csv(file_path+'valid.csv',sep=',').head(1000)

    print(np.shape(Train_DS))
    print(np.shape(Actual_DS))
    print(np.shape(Val_DS))

    ##----------------------------------------------------------------------------------------------------------------##
    Train_DS, Actual_DS, y =  Data_Munging(Train_DS,Actual_DS, Val_DS)

    #pred_Actual  = RFC_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)
    #pred_Actual  = XGB_Classifier(Train_DS, y, Actual_DS, Sample_DS, Grid=False)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys)