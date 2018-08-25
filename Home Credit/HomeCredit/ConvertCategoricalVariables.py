import pandas as pd

readData = ReadData(".","HomeCredit","sa","Pass@123")        

Bureau = readData.getData("[Final].[Bureau]")
BureauBalance = readData.getData("[Final].[BureauBalance]")
CreditCardBalance = readData.getData("[Final].[CreditCardBalance]")
InstallmentsPayments = readData.getData("[Final].[InstallmentsPayments]")
POSCashBalance = readData.getData("[Final].[POSCashBalance]")
PreviousApplication = readData.getData("[Final].[PreviousApplication]")

def convertCategoricalVaribalesToOneHotEncoding(data):
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
    
def getCategoricalAggregates(data,cols):
    final = pd.DataFrame()
    for col in cols:
        tmpData = data.groupby(["SK_ID_CURR"]).agg({col:["mean","sum"]})    
        tmpData.columns = ["_".join(x) for x in tmpData.columns.ravel()]
        final = pd.concat([tmpData,final],axis = 1)
    
    return final
        
        
Bureau_tmp,cols = convertCategoricalVaribalesToOneHotEncoding(Bureau)
Bureau_tmp = getCategoricalAggregates(Bureau_tmp,cols)

CreditCardBalance_tmp,cols = convertCategoricalVaribalesToOneHotEncoding(CreditCardBalance)
CreditCardBalance_tmp = getCategoricalAggregates(CreditCardBalance_tmp,cols)

InstallmentsPayments_tmp,cols = convertCategoricalVaribalesToOneHotEncoding(InstallmentsPayments)
InstallmentsPayments_tmp = getCategoricalAggregates(InstallmentsPayments_tmp,cols)

POSCashBalance_tmp,cols = convertCategoricalVaribalesToOneHotEncoding(POSCashBalance)
POSCashBalance_tmp = getCategoricalAggregates(POSCashBalance_tmp,cols)

PreviousApplication_tmp,cols = convertCategoricalVaribalesToOneHotEncoding(PreviousApplication)
PreviousApplication_tmp = getCategoricalAggregates(PreviousApplication_tmp,cols)

