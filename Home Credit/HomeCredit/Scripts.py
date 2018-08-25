# -*- coding: utf-8 -*-

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
    
readData = ReadData(".","HomeCredit","sa","Pass@123")

bureau = readData.getData("Final.Bureau")
readData.exploreData(bureau)
bureau.dtypes

prevApp = readData.getData("[Final].[PreviousApplication]")
readData.exploreData(prevApp)
prevApp.dtypes


AppData = readData.getData("[dbo].[ApplicationFullDataset]")

[reaDdata.checkPlotForNumericalFeatures(train[~np.isnan(train[column])][column]) for column in train.columns if train[column].dtype.kind in 'if']

readData.exploreCategoricalData(X_train,y_train)

