import pandas as pd
import pymssql as mssql
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
sys.path.append("F:\Sid\Learnings\Data Scientist\Machine Learning With Kaggle\KaggleProblems\Home Credit\HomeCredit")
from ReadData import ReadData
import pdb
import pickle

%matplotlib

class Models:
    def __init__(self):
        pass
    
    def addFeatures(self,train):
        train.loc[:,"MIN_EXT_SRC"] = train[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].min(axis = 1)
        train.loc[:,"MAX_EXT_SRC"] = train[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].max(axis = 1)
        train.loc[:,"MEAN_EXT_SRC"] = train[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].mean(axis = 1)
        train.loc[:,"SUM_EXT_SRC"] = train[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].sum(axis = 1)
        train.loc[:,"MED_EXT_SRC"] = train[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].median(axis = 1)
        return train
    
                
    def convertCategoricalVaribalesToOneHotEncoding(self,data):
        cat_vars = []
        original_columns = list(data.columns)
        for column in data.columns:        
            if data[column].dtype.kind in "O":
                cat_vars.append(column)               
                dummyCol = column + "_"                
                cat_list = pd.get_dummies(data[column], prefix=dummyCol)
                data1=data.join(cat_list)
                data=data1
        
        new_columns = [c for c in data.columns if c not in original_columns]
        data.drop(cat_vars,axis = 1,inplace = True)
        return data,new_columns
            
    def LogisticRegression(self,train_x,train_y,test_x,test_y,algorithm = ""):
        from sklearn.linear_model import LogisticRegression       
        
        model = LogisticRegression(class_weight = 'balanced')    
        model.fit(train_x,train_y)
        
        results = model.predict_proba(test_x)
        results = results[:,1]
        return model,results
        
    def AdaBoostClassifier(self,train_x,train_y,test_x,test_y,algorithm = ""):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        dt = DecisionTreeClassifier(class_weight = "balanced")
        model = AdaBoostClassifier(n_estimators=100, base_estimator=dt)
        
        model.fit(train_x,train_y)
        results = model.predict_proba(test_x)
        results = results[:,1]
        return model,results
    
    def lightgbm_tmp(self,train,test):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import classification_report as CR, roc_auc_score as ROC
        
        data_x = np.array(train.loc[:,train.columns != "TARGET"])
        data_y = np.array(train[["TARGET"]])
        data_test_x = np.array(test)
        
        skf = StratifiedKFold(n_splits=2)
        for train_index, test_index in skf.split(data_x, data_y):
            train_x, test_x = data_x[train_index], data_x[test_index]
            train_y, test_y = data_y[train_index], data_y[test_index]
        
            model,result = self.lightgbm(train_x,train_y,test_x,test_y)
            print("ROC",ROC(test_y,result))
        
    def lightgbm(self,train_x,train_y,test_x,test_y,algorithm = "gbdt"):
        import lightgbm
        best_iteration = 0
        best_leaves = 0
        #for leaves in range(2,50):
        
        model = lightgbm.LGBMClassifier(
                                    #n_estimators=100, silent=True, class_weight="balanced"
                                    #,num_leaves = 45
                                    #,boosting_type=algorithm
                                    #nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            class_weight="balanced",
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1             )
        
        if test_y == "":
            model.fit(train_x,train_y,eval_set=[(train_x, train_y)],eval_metric = 'auc')
        else:
            model.fit(train_x,train_y,eval_set=[(test_x, test_y)],eval_metric = 'auc')
        
        results = model.predict_proba(test_x)
        results = results[:,1]
        return model,results
    
    def RandomForestClassifier(self,train_x,train_y,test_x,test_y,algorithm = "gini"):
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=100,class_weight="balanced",oob_score=True,
                                       criterion = algorithm)
        model.fit(train_x,train_y)
        results = model.predict_proba(test_x)
        results = results[:,1]
        return model,results
    
    def KNN(self,train_x,train_y,test_x,test_y,algorithm='ball_tree'):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=6, algorithm=algorithm)
        model.fit(train_x,train_y)
        
        results = model.predict_proba(test_x)
        results = results[:,1]
        return model,results
    
    def DecisionTreeClassifier(self,train_x,train_y,test_x,test_y,algorithm = "gini"):
        from sklearn.tree import DecisionTreeClassifier
        
        model = DecisionTreeClassifier(class_weight="balanced",criterion = algorithm)
        model.fit(train_x,train_y)
        
        results = model.predict_proba(test_x)
        results = results[:,1]
        return model,results
    
    def XGBoostClassifier(self,train_x,train_y,test_x,test_y,algorithm = "gbtree"):
        from xgboost import XGBClassifier
        
        model = XGBClassifier(n_estimators=100,objective = "binary:logistic", eval_metric = "auc",
                              booster = algorithm)
        
        if test_y == "":
            model.fit(train_x,train_y,eval_set=[(train_x, train_y)],eval_metric = 'auc')
        else:
            model.fit(train_x,train_y,eval_set=[(test_x, test_y)],eval_metric = 'auc')
        
        results = model.predict_proba(test_x)
        results = results[:,1]
        return model,results
    
    def TensorFlowModel(self,train_x,train_y,test_x,test_y,algorithm = ""):
        import tensorflow as tf
        from tensorflow import keras
        
        model = keras.Sequential([
                keras.layers.Dense(32,activation = tf.nn.relu),
                keras.layers.Dense(64,activation = tf.nn.relu),
                keras.layers.Dense(32,activation = tf.nn.relu),
                keras.layers.Dense(1,activation = tf.nn.sigmoid),
                ])
        
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss = "binary_crossentropy",
                      metrics = [keras.metrics.categorical_accuracy])
        
        if test_y == "":
            model.fit(train_x,train_y, epochs=100,batch_size=1000,validation_data=(train_x,train_y),verbose=1)
        else:
            model.fit(train_x,train_y, epochs=100,batch_size=1000,validation_data=(test_x,test_y),verbose=1)
        
        results = model.predict_proba(test_x)
        return model,results[:,0]
    
    def ensembleStacking(self, train,test, iteration = 5):
        from sklearn.linear_model import LogisticRegression       
        from sklearn.metrics import classification_report as CR, roc_auc_score as ROC
        from sklearn.model_selection import StratifiedKFold
        model = 0
        roc = 0
        models = []
        
        algorithms = {
                      "xgboost":{"algo":["gbtree","gblinear","dart"] , "method":self.XGBoostClassifier},
                      "tf":{"algo":[""], "method": self.TensorFlowModel},
                      "rfc":{"algo":["gini","entropy"] , "method":self.RandomForestClassifier},
                      "lgbm":{"algo":["gbdt","dart","goss"] , "method": self.lightgbm},
                      "logit": {"algo":[""] , "method":self.LogisticRegression},
                      "ada":{"algo":[""] , "method":self.AdaBoostClassifier},
                      "dt":{"algo":["gini","entropy"] , "method":self.DecisionTreeClassifier}
                      
                      }
        
        skf = StratifiedKFold(n_splits=iteration)
        
        data_x = np.array(train.loc[:,train.columns != "TARGET"])
        data_y = np.array(train[["TARGET"]])
        data_test_x = np.array(test)
        
        df_val_pred = pd.DataFrame()
        df_test_pred = pd.DataFrame()
        index = 0
        target = np.array([], dtype = np.int32)
        
        for train_index, test_index in skf.split(data_x, data_y):
            train_x, test_x = data_x[train_index], data_x[test_index]
            train_y, test_y = data_y[train_index], data_y[test_index]
            
            df_val_pred_algo = pd.DataFrame()
            for key in algorithms: 
                algosPerClassifier = algorithms[key]["algo"]
                for algo in algosPerClassifier:
                
                    method = algorithms[key]["method"]
                    model,predict_prob = method(train_x,train_y,test_x,test_y,algorithm = algo)
                    
                    modelName = key + "_"+algo
                    
                    #---------Merge result of each algo
                    #pdb.set_trace()
                    df_val_pred_algo = pd.concat([df_val_pred_algo,
                                                 pd.DataFrame({modelName: predict_prob})],
                                             axis = 1)
                    
                    if index == 0:
                        #-------Do prediction for entire dataset i.e. train and test
                        #-------This will be your test set
                        model,result = method(data_x, data_y,data_test_x,"")
                        df_test_pred = pd.concat([df_test_pred,pd.DataFrame({modelName: result})],axis = 1)
            
            #--------Add target variable
            target = np.concatenate((target,test_y[:,0]), axis = 0)
            
            #-------Merge all the splits row wise
            df_val_pred = pd.concat([df_val_pred,df_val_pred_algo],axis = 0)
            index = index + 1
            
        df_val_pred = np.array(df_val_pred)
        df_test_pred = np.array(df_test_pred)
        pdb.set_trace()
        stackingModel = LogisticRegression(class_weight = 'balanced')    
        stackingModel.fit(df_val_pred,target)
        
        #--------Metrics info on training set
        predict_prob = stackingModel.predict_proba(df_val_pred)[:,1]
        roc = ROC(target,predict_prob)
        print("ROC %f" % roc)
        
        predict_prob = stackingModel.predict_proba(df_test_pred)[:,1]
        
        return predict_prob,df_val_pred,target,df_test_pred

    def evaluateModel(self,test_x,test_y,model):
        from sklearn.metrics import classification_report as CR, roc_auc_score as ROC
        
        predict_result = model.predict(test_x)
        predict_prob = model.predict_proba(test_x)[:,1]
        
        
        crReport = CR(test_y,predict_result)
        acc = model.score(test_x,test_y)
        roc = ROC(test_y,predict_prob)
        #print("Accuracy %f" % model.score(test_x,test_y))        
        return crReport,acc,roc
    
    def getScalingColumns(self,featureSet):
        columns=[]
        for column in featureSet.columns:
            if featureSet[column].dtype.kind in "if":
                columns.append(column)
        return columns
        
    def scaleDataset(self, train,test,scalingColumns):
        from sklearn.preprocessing import MinMaxScaler 
        scalingColumns = [col for col in scalingColumns if col in test.columns]
        nonScalingColumns = [col for col in test.columns if col not in scalingColumns]
        fullColumnList = scalingColumns + nonScalingColumns
        
        scaler = MinMaxScaler().fit(train[scalingColumns])
        
        train_scaled = pd.DataFrame(scaler.transform(train[scalingColumns]),columns = scalingColumns)
        train_nonScaled = train[nonScalingColumns]
        
        test_scaled = pd.DataFrame(scaler.transform(test[scalingColumns]),columns = scalingColumns)
        test_nonScaled = pd.DataFrame(test[nonScalingColumns])
        tmp = [col for col in train_scaled.columns if col in train_nonScaled.columns]
        #pdb.set_trace()
        train = pd.concat([train_scaled,train_nonScaled,train[["TARGET"]]],axis = 1,ignore_index = True)
        test = pd.concat([test_scaled,test_nonScaled],axis = 1,ignore_index = True)
        
        train.columns = fullColumnList + ["TARGET"]
        test.columns = fullColumnList
        
        return train,test
        
    def trainDataset(self,train, trainingAlgo, test_size=.33, iteration = 10):
        from sklearn.model_selection import StratifiedShuffleSplit
        model = 0
        roc = 0
        
        skf = StratifiedShuffleSplit(n_splits=iteration,test_size = .33)
        data_x = np.array(train.loc[:,train.columns != "TARGET"])
        data_y = np.array(train[["TARGET"]])
        
        for train_index, test_index in skf.split(data_x, data_y):
        #for index in range(iteration):
            X_train, X_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
            #X_train, X_test, y_train, y_test = dataSplitFn(train
             #                                              ,test_size=test_size)
            #--------DROP ID column from train and test
            tmpModel,result = trainingAlgo(X_train,y_train,X_test)
               
            model = tmpModel
            """
            _,acc,rocScore = models.evaluateModel(X_test,y_test,tmpModel)
            if roc < rocScore:
                roc = rocScore
                model = tmpModel
                print("Accurachy %f, ROC Score %f" % (acc,roc))
            """
        return model

#------Get feature set and create classes   
readData = ReadData(".","HomeCredit","sa","Pass@123")        
models = Models()

featureSet = readData.getData("dbo.FeatureSet")
featureSet = models.addFeatures(featureSet)
scalingColumns = models.getScalingColumns(featureSet)
featureSet, _ = models.convertCategoricalVaribalesToOneHotEncoding(featureSet)

train = featureSet[featureSet["TARGET"] != -1].reset_index(drop = True)
test = featureSet[featureSet["TARGET"] == -1].reset_index(drop = True)

test_ids = test["SK_ID_CURR"]

test.drop(["TARGET","SK_ID_CURR"],axis = 1,inplace = True)
train.drop(["SK_ID_CURR"],axis = 1,inplace = True)
train["TARGET"] = train["TARGET"].astype("category")

train,test = models.scaleDataset(train,test,scalingColumns) 

train.columns[train.isna().any()]
#----------Balncing the dataset reducing the performance
#train_bal = models.balanceDataset(train)

"""
            ROC AUC
Linear Regression: .67
LightGBM: .769
Random Forest: .74
AdaBoost: .54
KNN: .52
Ensembling With Stacking: .774
Ensembling With Ranking: 
Ensembling With Averaging:  .74
Ensembling with Stacking (NN): 776
"""
#modelWithResultSet,df = models.trainDataset(train
 #       ,trainingAlgo=models.ensembleStacking
  #      ,iteration=1
   #     )

model,result = models.lightgbm_tmp(train,test)
result,df_val_pred,target,df_test_pred = models.ensembleStacking(train,test,5)
result = result.reshape(-1,1)

tmpResult = pd.DataFrame({"SK_ID_CURR":test_ids,"TARGET":result[:,0]})
fileName = "F:\Sid\Learnings\Data Scientist\Machine Learning With Kaggle\KaggleProblems\Home Credit\HomeCredit\Data\\result.csv"
tmpResult.to_csv(fileName,sep = ",",index = False)

df_val_fileName = "F:\Sid\Learnings\Data Scientist\Machine Learning With Kaggle\KaggleProblems\Home Credit\HomeCredit\Data\\df_val_pred.npy"
df_test_fileName = "F:\Sid\Learnings\Data Scientist\Machine Learning With Kaggle\KaggleProblems\Home Credit\HomeCredit\Data\\df_test_pred.npy"
target_fileName = "F:\Sid\Learnings\Data Scientist\Machine Learning With Kaggle\KaggleProblems\Home Credit\HomeCredit\Data\\target.npy"
"""
Save to disk
"""
np.save(df_val_fileName,df_val_pred)
np.save(df_test_fileName,df_test_pred)
np.save(target_fileName,target)

"""
Load from disk
"""
df_val_pred = np.load(df_val_fileName)
df_test_pred = np.load(df_test_fileName)
target = np.load(target_fileName)


def metaLearner(data_x,data_y,test,models):
    from sklearn import svm
    from sklearn.metrics import classification_report as CR, roc_auc_score as ROC
    from sklearn.model_selection import StratifiedKFold
    
    
    roc_tmp = 0
    model_final = 0
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(data_x, data_y):
        train_x, test_x = data_x[train_index], data_x[test_index]
        train_y, test_y = data_y[train_index], data_y[test_index]
        model,result = models(train_x,train_y,test_x,test_y)
            
        roc = ROC(test_y,result)
        print("ROC:",roc)
        if roc_tmp  < roc:
            roc_tmp = roc
            model_final = model
    
    pdb.set_trace()
    result = model_final.predict_proba(test)
    return result

result = metaLearner(df_val_pred,target,df_test_pred,models.RandomForestClassifier)
result = result[:,1]



result = modelWithResultSet.predict_proba(test)

feats = {} # a dict to hold feature_name: feature_importance
columns = train.columns[train.columns != "TARGET"]
for feature, importance in zip(columns, model.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').to_csv("F:\Features.csv",sep = ",")

#-------------------------Read and write model object

import pickle
modelFileName = "F:\Sid\Learnings\Data Scientist\Machine Learning With Kaggle\Home Credit\HomeCredit\model.pkl"
with open(modelFileName, "wb") as f:
    pickle.dump(modelWithResultSet,f,pickle.HIGHEST_PROTOCOL)

with open(modelFileName, "rb") as f:
    modelWithResultSet = pickle.load(f)