'''
Created on March 10 2021

@auther: Jiachen Yan
'''



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data= pd.read_csv(r"C:\Users\xiah\Desktop\data_est\moe\moe_descriptor_std.csv")


# X = np.asarray(data.iloc[0:154,4:], dtype='float32')
# Y = np.asarray(data.iloc[0:154,3], dtype='float32')
#
# scaler =StandardScaler()
# scaler.fit(X)
# st_data =scaler.transform(X)
# st_data_out =np.asarray(st_data,dtype='float32')
# print(st_data_out)


'''
Normalized data
'''

Xtrain = data.iloc[0:122,4:]
Ytrain = data.iloc[0:122,3]
Xtest = data.iloc[122:154,4:]
Ytest = data.iloc[122:154,3]

'''
5-fold cross validation
'''

SEED = 1

rfc = RandomForestRegressor(random_state=SEED)
CV_score = cross_val_score(rfc, Xtrain, Ytrain, cv=5, scoring='r2').mean()
CV_predictions = cross_val_predict(rfc, Xtrain, Ytrain, cv=5)
print(CV_score)
expvspred_5cv = {'Exp': Ytrain, 'Pred':CV_predictions}
pd.DataFrame(expvspred_5cv).to_excel('Random_forest_5fcv_predictions.xlsx')

regressor = rfc.fit(Xtrain, Ytrain)
test_predictions = regressor.predict(Xtest)
score_test = regressor.score(Xtest,Ytest)
print(score_test)
expvspred_test = {'Exp':Ytest,'Pred':test_predictions}
pd.DataFrame(expvspred_test).to_excel('Random_forest_test_predictions.xlsx')



