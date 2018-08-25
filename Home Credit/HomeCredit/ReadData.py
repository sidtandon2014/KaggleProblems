import pandas as pd
import pymssql as mssql
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

matplotlib.style.use('ggplot')

class ReadData:
    def __init__(self,serverName,dbName,userName,password):
        self.con = mssql.connect(server = serverName,database = dbName,user= userName,password = password)    
                
    def __del__(self):
        self.con.close()
    
    def dbCommit(self):
        self.con.commit()
        
    def getData(self,viewName):
        stmt = "SELECT * FROM " + viewName
        prevApp = pd.read_sql(stmt,self.con)
        for column in prevApp.columns:
            if prevApp[column].dtype.kind not in 'if':
                prevApp[column] = prevApp[column].astype("category")
                
        return prevApp

    def exploreData(self,data):
        [self.checkPlotForNumericalFeatures(data[~np.isnan(data[column])][column]) for column in data.columns 
        if (data[column].dtype.kind in 'if')]
        
    def exploreCategoricalData(self,data,y):
        [self.checkPlotForCategoricalVariables(data[column],y) for column in data.columns 
        if (data[column].dtype.kind in 'O')]
            
    def checkPlotForNumericalFeatures(self,columnData,bins = 'auto'):
        if columnData.dtype.kind in 'if':
            fig = plt.figure()        
            subplot = fig.add_subplot(111)                 
            subplot.set_title(columnData.name) 
            subplot.hist(columnData,bins = bins)
            plt.show()
    
    def checkPlotForCategoricalVariables(self,columnData,y):
        pd.crosstab(columnData,y).plot(kind = 'bar')
        plt.xlabel(columnData.name)
        plt.ylabel("TARGET")
    
    def getModelForAnnuity(self,data):
        from sklearn.linear_model import LinearRegression 
        import pdb
        pdb.set_trace()
        train = data[~np.isnan(data["AMT_ANNUITY"])]
        test =  data[np.isnan(data["AMT_ANNUITY"])]["AMT_CREDIT"]
        
        train_x = train["AMT_CREDIT"]
        train_y = train["AMT_ANNUITY"]
        
        train_x = train_x.reshape(train_x.shape[0],1)
        train_y = train_y.reshape(train_x.shape[0],1)
        test = test.reshape(test.shape[0],1)
        
        model = LinearRegression()
        model.fit(train_x,train_y)
        result = model.predict(test)
        return result,model
        
