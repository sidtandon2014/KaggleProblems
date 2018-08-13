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
        
        return model,""
        
    def AdaBoostClassifier(self,train_x,train_y,test_x,test_y,algorithm = ""):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        dt = DecisionTreeClassifier(class_weight = "balanced")
        model = AdaBoostClassifier(n_estimators=100, base_estimator=dt)
        
        model.fit(train_x,train_y)
        return model,""
            
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
        return model,""
    
    def RandomForestClassifier(self,train_x,train_y,test_x,test_y,algorithm = "gini"):
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=100,class_weight="balanced",oob_score=True,
                                       criterion = algorithm)
        model.fit(train_x,train_y)
        
        return model,""
    
    def KNN(self,train_x,train_y,test_x,test_y,algorithm='ball_tree'):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=6, algorithm=algorithm)
        model.fit(train_x,train_y)
        
        return model,""
    
    def DecisionTreeClassifier(self,train_x,train_y,test_x,test_y,algorithm = "gini"):
        from sklearn.tree import DecisionTreeClassifier
        
        model = DecisionTreeClassifier(class_weight="balanced",criterion = algorithm)
        model.fit(train_x,train_y)
        
        return model,""
    
    def XGBoostClassifier(self,train_x,train_y,test_x,test_y,algorithm = "gbtree"):
        from xgboost import XGBClassifier
        
        model = XGBClassifier(n_estimators=100,objective = "binary:logistic", eval_metric = "auc",
                              booster = algorithm)
        model.fit(train_x,train_y,eval_set=[(test_x, test_y)],eval_metric = 'auc')
        
        return model,""
    
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
        
        model.fit(train_x,train_y, epochs=100,batch_size=1000,validation_data=(test_x,test_y),verbose=1)
        print(model.evaluate(test_x,test_y))
        
        return model,""
        
    def H2OModel(self,train_x,train_y,test_x,test_y,algorithm = "gbtree"):
        import h2o
        from h2o.estimators.deeplearning import H2ODeepLearningEstimator
        h2o.init()
        h2o.connect()
        train = pd.concat((pd.DataFrame(train_x),pd.Series(train_y.reshape(-1))), axis = 1)
        test = pd.concat((pd.DataFrame(test_x),pd.Series(test_y.reshape(-1))),axis = 1)
        
        totalColumns = len(train.columns)
        columns = ["C_"+ str(x) for x in range(totalColumns)]
        train.columns = columns
        test.columns = columns
        
        train_fr = h2o.H2OFrame(train,column_names = None)
        test_fr = h2o.H2OFrame(test,column_names = None)
        
        train_fr["C_415"] = train_fr["C_415"].asfactor()
        test_fr["C_415"] = test_fr["C_415"].asfactor()
        model = H2ODeepLearningEstimator(
                loss = "cross_entropy"
                ,stopping_metric = "auc"
                ,hidden=[32,32,32]
                ,epochs = 1000  
                ,stopping_rounds = 2
                ,distribution = "bernoulli"
                ,activation = "tanh_with_dropout"
                )
        model.train(x=train_fr.col_names[0:totalColumns-1], y= train_fr.col_names[totalColumns-1]
                    , training_frame=train_fr
                    , validation_frame= test_fr)
        
        return model,""
    
    def catBoostModel(self,train_x,train_y,test_x,test_y,algorithm = "gbtree"):
        from catboost import ca
    
    def ensemblingWithRanking(self,models,test):
        #---------Normalized scores
        scores = 0
        normalizedScores = {}
        resultSet = np.array([])   
        for items in models:
            tmpKey = [key for key in items.keys()][0]
            
            if tmpKey != "StackingModel":
                tmpValue = [value for value in items.values()][0][1]
                scores += tmpValue
                normalizedScores[tmpKey] = tmpValue
        
        for items in models:
            tmpKey = [key for key in items.keys()][0]
            if tmpKey != "StackingModel":
                tmpModel = [value for value in items.values()][0][0]
                
                normalizedScores[tmpKey] = normalizedScores[tmpKey]/ scores
                
                resultSet_tmp = tmpModel.predict_proba(test)[:,1] * normalizedScores[tmpKey]
                resultSet_tmp = resultSet_tmp.reshape(-1,1)
                
                if len(resultSet) == 0:
                    resultSet = resultSet_tmp
                else:
                    resultSet = np.concatenate((resultSet,resultSet_tmp),axis = 1)
                    
        pdb.set_trace()
        return resultSet.sum(axis = 1)
    
    def ensemblingWithAveraging(self,models,test):
        #---------Normalized scores
        resultSet = np.array([])   
        
        for items in models:
            tmpKey = [key for key in items.keys()][0]
            if tmpKey != "StackingModel":
                tmpModel = [value for value in items.values()][0][0]
                
                resultSet_tmp = tmpModel.predict_proba(test)[:,1]
                resultSet_tmp = resultSet_tmp.reshape(-1,1)
                
                if len(resultSet) == 0:
                    resultSet = resultSet_tmp
                else:
                    resultSet = np.concatenate((resultSet,resultSet_tmp),axis = 1)
                    
        return resultSet.mean(axis = 1)

    def ensemblingWithLogit(self,models,test):
        #---------Normalized scores
        resultSet = np.array([])   
        
        for items in models:
            tmpKey = [key for key in items.keys()][0]
            tmpModel = [value for value in items.values()][0][0]
            
            if tmpKey != "StackingModel":
                resultSet_tmp = tmpModel.predict_proba(test)[:,1]
                resultSet_tmp = resultSet_tmp.reshape(-1,1)
                
                if len(resultSet) == 0:
                    resultSet = resultSet_tmp
                else:
                    resultSet = np.concatenate((resultSet,resultSet_tmp),axis = 1)
            else:
                stackingModel = tmpModel
                
        return stackingModel.predict_proba(resultSet)[:,1]
    
    def ensembleStacking(self, train_x,train_y,test_x,test_y):
        from sklearn.linear_model import LogisticRegression       
        from sklearn.metrics import classification_report as CR, roc_auc_score as ROC
        
        algorithms = {"rfc":{"algo":["gini","entropy"] , "method":self.RandomForestClassifier},
                      "tf":{"algo":[""], "method": self.TensorFlowModel},
                      "lgbm":{"algo":["gbdt","dart","goss"] , "method": self.lightgbm},
                      "logit": {"algo":[""] , "method":self.LogisticRegression},
                      "ada":{"algo":[""] , "method":self.AdaBoostClassifier},
                      "dt":{"algo":["gini","entropy"] , "method":self.DecisionTreeClassifier},
                      "xgboost":{"algo":["gbtree","gblinear","dart"] , "method":self.XGBoostClassifier},
                      }
        
        models = []
        df_val_pred = pd.DataFrame()
        for key in algorithms: 
            algosPerClassifier = algorithms[key]["algo"]
            for algo in algosPerClassifier:
                method = algorithms[key]["method"]
                model,_ = method(train_x,train_y,test_x,test_y,algorithm = algo)
                result_tmp = model.predict_proba(test_x)
                if len(result_tmp.shape) == 2:
                    predict_prob = result_tmp[:,1]
                else:
                    predict_prob = result_tmp
                    
                modelName = key + "_"+algo
                df_val_pred = pd.concat([df_val_pred,pd.DataFrame(predict_prob,columns = [modelName])],
                                         axis = 1)
                models.append({modelName:(model,ROC(test_y,predict_prob))})
        
        df_val_pred = np.array(df_val_pred)
        
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
        pdb.set_trace()
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
featureSet = models.addFeatures(featureSet)
scalingColumns = models.getScalingColumns(featureSet)
featureSet = models.convertCategoricalVaribalesToOneHotEncoding(featureSet)

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
modelWithResultSet,df = models.trainDataset(train
        ,trainingAlgo=models.ensembleStacking
        ,iteration=1
        )

result = models.ensemblingWithLogit(modelWithResultSet,np.array(test))
result = result.reshape(-1,1)

tmpResult = pd.DataFrame({"SK_ID_CURR":test_ids,"TARGET":result[:,0]})
fileName = "F:\Sid\Learnings\Data Scientist\Machine Learning With Kaggle\Home Credit\HomeCredit\Data\\result.csv"
tmpResult.to_csv(fileName,sep = ",",index = False)

result = modelWithResultSet.predict_proba(test)

feats = {} # a dict to hold feature_name: feature_importance
columns = train.columns[train.columns != "TARGET"]
for feature, importance in zip(columns, model.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').to_csv("F:\Features.csv",sep = ",")

h2o.

#-------------------------Read and write model object

import pickle
modelFileName = "F:\Sid\Learnings\Data Scientist\Machine Learning With Kaggle\Home Credit\HomeCredit\model.pkl"
with open(modelFileName, "wb") as f:
    pickle.dump(modelWithResultSet,f,pickle.HIGHEST_PROTOCOL)

with open(modelFileName, "rb") as f:
    modelWithResultSet = pickle.load(f)