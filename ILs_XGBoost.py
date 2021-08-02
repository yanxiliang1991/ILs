'''
Created on March 10 2021

@auther: Jiachen Yan
'''



from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Data normalization
'''

data= pd.read_csv("moe_descriptors.csv")


X = np.asarray(data.iloc[0:154,4:], dtype='float32')
Y = np.asarray(data.iloc[0:154,3], dtype='float32')

scaler =StandardScaler()
scaler.fit(X)
st_data =scaler.transform(X)



Xtrain = st_data[0:122]
Ytrain = data.iloc[0:122,3]
Xtest = st_data[122:154]
Ytest = data.iloc[122:154,3]


'''
5-fold cross validation
'''

SEED =0

XGB = XGBR(n_estimators=100,random_state=0)
CV_score = cross_val_score(XGB, Xtrain, Ytrain, cv=5,scoring='r2').mean()
CV_predictions = cross_val_predict(XGB, Xtrain, Ytrain, cv=5)
print(CV_score)
expvspred_5cv = {'Exp': Ytrain, 'Pred':CV_predictions}
pd.DataFrame(expvspred_5cv).to_excel('XGBoost_5fcv_predictions.xlsx')

regressor = XGB.fit(Xtrain, Ytrain)
test_predictions = regressor.predict(Xtest)
score_test = regressor.score(Xtest,Ytest)
print(score_test)
expvspred_test = {'Exp':Ytest,'Pred':test_predictions}
pd.DataFrame(expvspred_test).to_excel('XGBoost_test_predictions.xlsx')