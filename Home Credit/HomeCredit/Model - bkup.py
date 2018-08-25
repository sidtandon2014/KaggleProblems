import pandas as pd
import pymssql as mssql
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
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
        
    def performStratifiedSplit(self,train,test_size = .33,targetVariable = "TARGET"):
        
        train.loc[:,"ID"] = range(0,len(train))
        
        totalRows = len(train)
        train_0 = train[train[targetVariable] == 0]        
        train_1 = train[train[targetVariable] == 1]
        
        train_0_Per = len(train_0) / totalRows
        train_1_Per = len(train_1) / totalRows
        
        train_split_1 = math.floor((1-test_size) * train_1_Per * totalRows)
        train_split_0 = math.floor((1-test_size) * train_0_Per * totalRows)
        
        #-------Allthe Ids with 1 and 0 separation
        train_Ids_1 = train[train[targetVariable] == 1]["ID"]
        train_Ids_0 = train[train[targetVariable] == 0]["ID"]
        
        #-------------Get SK_ID_CURR for specific numbe rof samples
        train_split_Ids_1 = train_Ids_1.sample(train_split_1)
        train_split_Ids_0 = train_Ids_0.sample(train_split_0)
        
        #-------Get only those Ids which are not part of train_split_Ids_1
        tmp = pd.merge(train_Ids_1.to_frame(),train_split_Ids_1.to_frame(), on = ["ID"],how = "left",indicator = True)
        test_split_Ids_1 = tmp[tmp["_merge"] == 'left_only'][["ID"]]
        
        #-------Get only those Ids which are not part of train_split_Ids_0
        tmp = pd.merge(train_Ids_0.to_frame(),train_split_Ids_0.to_frame(), on = ["ID"],how = "left",indicator = True)
        test_split_Ids_0 = tmp[tmp["_merge"] == 'left_only'][["ID"]]
        
        #------Merge 0 and 1 Ids together
        train_Ids = pd.concat([train_split_Ids_1,train_split_Ids_0])
        test_Ids = pd.concat([test_split_Ids_1,test_split_Ids_0])
        
        #-----Get complete dataframe based on train_ids
        tmp = pd.merge(train,train_Ids.to_frame(),on = ["ID"],how = "inner")
        train_x = tmp.loc[:,tmp.columns != targetVariable]
        train_y = tmp.loc[:,tmp.columns == targetVariable]        
        train_x.drop(["ID"],axis = 1,inplace = True)
        
        #-----Get complete dataframe based on test_ids
        tmp = pd.merge(train,test_Ids,on = ["ID"],how = "inner")
        test_x = tmp.loc[:,tmp.columns != targetVariable]
        test_y = tmp.loc[:,tmp.columns == targetVariable]
        test_x.drop(["ID"],axis = 1,inplace = True)
        
        return train_x,test_x,train_y,test_y
    
    def balanceDataset(self,train):
        from imblearn.over_sampling import ADASYN
        
        ada = ADASYN(random_state=10, ratio="minority")
        x = train.loc[:,train.columns != "TARGET"]
        y = train.loc[:,train.columns == "TARGET"]
        
        #pdb.set_trace()
        X,Y = ada.fit_sample(x,y)
        
        tmpDs = pd.concat(
                [pd.DataFrame(X,columns = x.columns),pd.DataFrame(Y,columns = y.columns)]
                ,axis = 1)
        return tmpDs
                
    def convertCategoricalVaribalesToOneHotEncoding(self,data):
        cat_vars = []
        for column in data.columns:        
            if data[column].dtype.kind in "O":
                cat_vars.append(column)               
                dummyCol = column + "_"                
                cat_list = pd.get_dummies(data[column], prefix=dummyCol)
                data1=data.join(cat_list)
                data=data1
                
        data.drop(cat_vars,axis = 1,inplace = True)
        return data
            
    def LogisticRegression(self,train_x,train_y,test_x,test_y,algorithm = ""):
        from sklearn.linear_model import LogisticRegression       
        
        model = LogisticRegression(class_weight = 'balanced')    
        result = model.fit(train_x,train_y)
        
        return model
        
    def AdaBoostClassifier(self,train_x,train_y,test_x,test_y,algorithm = ""):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        dt = DecisionTreeClassifier(class_weight = "balanced")
        model = AdaBoostClassifier(n_estimators=100, base_estimator=dt)
        
        model.fit(train_x,train_y)
        return model
            
    def lightgbm(self,train_x,train_y,test_x,test_y,algorithm = "gbdt"):
        import lightgbm
        best_iteration = 0
        best_leaves = 0
        #for leaves in range(2,50):
            
        model = lightgbm.LGBMClassifier(n_estimators=100, silent=True, class_weight="balanced"
                                    ,num_leaves = 45
                                    ,boosting_type=algorithm
                                    ,)
        model.fit(train_x,train_y,eval_set=[(test_x, test_y)],eval_metric = 'auc')
        
        """
        if best_iteration < model.best_score_["valid_0"]["auc"]:
            best_leaves = leaves
            best_iteration = model.best_score_["valid_0"]["auc"]
        """ 
        #print("Leaves %i AUC %f" % (best_leaves,best_iteration))
        return model
    
    def RandomForestClassifier(self,train_x,train_y,test_x,test_y,algorithm = "gini"):
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=100,class_weight="balanced",oob_score=True,
                                       criterion = algorithm)
        model.fit(train_x,train_y)
        
        return model
    
    def KNN(self,train_x,train_y,test_x,test_y,algorithm='ball_tree'):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=6, algorithm=algorithm)
        model.fit(train_x,train_y)
        
        return model
    
    def DecisionTreeClassifier(self,train_x,train_y,test_x,test_y,algorithm = "gini"):
        from sklearn.tree import DecisionTreeClassifier
        
        model = DecisionTreeClassifier(class_weight="balanced",criterion = algorithm)
        model.fit(train_x,train_y)
        
        return model
    
    def XGBoostClassifier(self,train_x,train_y,test_x,test_y,algorithm = "gbtree"):
        from xgboost import XGBClassifier
        
        model = XGBClassifier(n_estimators=100,objective = "binary:logistic", eval_metric = "auc",
                              booster = algorithm)
        model.fit(train_x,train_y,eval_set=[(test_x, test_y)],eval_metric = 'auc')
        
        return model
    
    def ensemblingWithRanking(self,models,test):
        #---------Normalized scores
        scores = 0
        normalizedScores = {}
        resultSet = np.array()
        for items in models:
            scores += items.value(1)
            normalizedScores[items.key] = items.value(1)
            
        for items in models:
            normalizedScores[items.key][1] = normalizedScores[items.key][1]/ scores
            resultSet = np.concatenate((resultSet,items.value[0].predict_proba(test_x)[:,1] * normalizedScores[items.key][1]),
                                       axis = 1)
        resultSet.sum(axis = 1)
        return resultSet.sum(axis = 1)
    
    def ensemblingWithAveraging(self,models,test):
        #---------Normalized scores
        pdb.set_trace()
        scores = 0
        normalizedScores = {}
        resultSet = np.array()
            
        for items in models:
            normalizedScores[items.key][1] = normalizedScores[items.key][1]/ scores
            resultSet = np.concatenate((resultSet,items.value[0].predict_proba(test_x)[:,1] / len(models)),
                                       axis = 1)
        resultSet.sum(axis = 1)
        return resultSet.sum(axis = 1)

    def ensemblingWithLogit(self,models,test):
        #---------Normalized scores
        resultSet = np.array()
            
        stackingModel = ""
        for items in models:
            if items.key != "StackingModel":
                resultSet = np.concatenate((resultSet,items.value[0].predict_proba(test_x)[:,1]),
                                       axis = 1)
            else:
                stackingModel = items.value[0]
                
        return stackingModel.predict_proba(test_x)[:,1]
    
    def ensembleStacking(self, train_x,train_y,test_x,test_y):
        from sklearn.linear_model import LogisticRegression       
        from sklearn.metrics import classification_report as CR, roc_auc_score as ROC
        
        algorithms = {"rfc":{"algo":["gini","entropy"] , "method":self.RandomForestClassifier},
                      "lgbm":{"algo":["gbdt","dart","goss"] , "method": self.lightgbm},
                      "logit": {"algo":[""] , "method":self.LogisticRegression},
                      "ada":{"algo":[""] , "method":self.AdaBoostClassifier},
                      "dt":{"algo":["gini","entropy"] , "method":self.DecisionTreeClassifier},
                      "knn":{"algo":["auto"] , "method":self.KNN},
                      "xgboost":{"algo":["gbtree","gblinear","dart"] , "method":self.XGBoostClassifier},
                      }
        
        models = []
        df_val_pred = pd.DataFrame()
        for key in algorithms: 
            algosPerClassifier = algorithms[key]["algo"]
            for algo in algosPerClassifier:
                method = algorithms[key]["method"]
                model = method(train_x,train_y,test_x,test_y,algorithm = algo)
                predict_prob = model.predict_proba(test_x)[:,1]
                modelName = key + "_"+algo
                df_val_pred = pd.concat([df_val_pred,pd.DataFrame(predict_prob,columns = [modelName])],
                                         axis = 1)
                models.append({modelName:(model,ROC(test_y,predict_prob))})
        
        df_val_pred = np.array(df_val_pred)
        
        pdb.set_trace()
        stackingModel = LogisticRegression(class_weight = 'balanced')    
        stackingModel.fit(df_val_pred,test_y)
        
        predict_prob = stackingModel.predict_proba(df_val_pred)[:,1]
        roc = ROC(test_y,predict_prob)
        print("ROC %f" % roc)
        models.append({"StackingModel":(stackingModel,roc)})
        return models,df_val_pred

    def evaluateModel(self,test_x,test_y,model):
        from sklearn.metrics import classification_report as CR, roc_auc_score as ROC
        
        predict_result = model.predict(test_x)
        predict_prob = model.predict_proba(test_x)[:,1]
        
        
        crReport = CR(test_y,predict_result)
        acc = model.score(test_x,test_y)
        roc = ROC(test_y,predict_prob)
        #print("Accuracy %f" % model.score(test_x,test_y))        
        return crReport,acc,roc
    
    def scaleDataset(self, train):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train)
        return scaler
        
    def trainDataset(self,train, trainingAlgo,dataSplitFn, ISTRAIN = 1, test_size=.33, iteration = 10):
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
            #if ISTRAIN == 1:
            tmpModel,df = trainingAlgo(X_train,y_train,X_test,y_test)
            model = tmpModel
            """
            _,acc,rocScore = models.evaluateModel(X_test,y_test,tmpModel)
            if roc < rocScore:
                roc = rocScore
                model = tmpModel
                print("Accurachy %f, ROC Score %f" % (acc,roc))
            """
        return model,df

#------Get feature set and create classes   
readData = ReadData(".","HomeCredit","sa","Pass@123")        
models = Models()

featureSet = readData.getData("dbo.FeatureSet")

featureSet = models.convertCategoricalVaribalesToOneHotEncoding(featureSet)
featureSet = models.addFeatures(featureSet)

train = featureSet[featureSet["TARGET"] != -1]
test = featureSet[featureSet["TARGET"] == -1]

test_ids = test["SK_ID_CURR"]

test.drop(["TARGET","SK_ID_CURR"],axis = 1,inplace = True)
train.drop(["SK_ID_CURR"],axis = 1,inplace = True)
train["TARGET"] = train["TARGET"].astype("category")

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
Ensembling With Stacking:.776
"""
model,scaler,df = models.trainDataset(train
    ,trainingAlgo=models.ensembleStacking
    ,dataSplitFn=models.performStratifiedSplit
    ,iteration=1)

test = scaler.transform(test)


#-----------------


result = model.predict_proba(test)

tmpResult = pd.DataFrame({"SK_ID_CURR":test_ids,"TARGET":result[:,1]})
fileName = "F:\Sid\Learnings\Data Scientist\Machine Learning With Kaggle\Home Credit\HomeCredit\Data\\result.csv"
tmpResult.to_csv(fileName,sep = ",",index = False)


feats = {} # a dict to hold feature_name: feature_importance
columns = train.columns[train.columns != "TARGET"]
for feature, importance in zip(columns, model.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').to_csv("F:\Features.csv",sep = ",")
