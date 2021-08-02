'''
Created on March 10 2021

@auther: Jiachen Yan
'''

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler



'''
Data normalization
'''

data= pd.read_csv("moe_descriptors.csv")

X = np.asarray(data.iloc[0:154,4:], dtype='float32')
Y = np.asarray(data.iloc[0:154,3], dtype='float32')

scaler =StandardScaler()
scaler.fit(X)
st_data =scaler.transform(X)

train_data = st_data[0:122]
train_targets = data.iloc[0:122,3]
test_data = st_data[122:154]
test_targets = data.iloc[122:154,3]


'''
Build the model
'''

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

'''
five-fold cross validation
'''

num_epochs = 300
y_pred_5cv = []
y_exp_5cv = []

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error

kf = KFold(n_splits=5, shuffle=True)
i = 0


for train_index, test_index in kf.split(train_data):
    partial_train_data, val_data = train_data[train_index], train_data[test_index]
    partial_train_targets, val_targets = train_targets[train_index], train_targets[test_index]

    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1)

    y_pred_5cv.extend(model.predict(val_data))
    y_exp_5cv.extend(val_targets)

    model.save('ANN_5cv{}.h5'.format(i))
    i += 1

y_exp_5cv = np.asarray(y_exp_5cv).reshape((len(train_data), ))
y_pred_5cv = np.asarray(y_pred_5cv).reshape((len(train_data), ))
print(y_exp_5cv)
print(y_pred_5cv)
print(r2_score(y_exp_5cv, y_pred_5cv))
expvspred_5cv = {'Exp': y_exp_5cv, 'Pred': y_pred_5cv}
pd.DataFrame(expvspred_5cv).to_excel('ExpvsPred_ANN_5CV.xlsx')


import matplotlib.pyplot as plt

plt.switch_backend('agg')
fig, ax = plt.subplots()


ax.scatter(y_exp_5cv, y_pred_5cv)

lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]

ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Exp.')
ax.set_ylabel('Pred.')
fig.savefig('ann.png',dpi=300)

'''
#Test set validation
'''

model_test = build_model()
history = model_test.fit(train_data, train_targets, epochs=num_epochs, batch_size=1,validation_data=(test_data,test_targets))
model_test.save('ANN_test.h5')
y_train_5cv = model_test.predict(train_data)
y_test_pred = model_test.predict(test_data)
y_test_pred = np.asarray(y_test_pred).reshape(len(test_data), )
test_loss = mean_squared_error(test_targets, y_test_pred)
print(r2_score(test_targets, y_test_pred))
print(test_loss)

expvspred_test = {'Exp': test_targets, 'Pred': y_test_pred}
pd.DataFrame(expvspred_test).to_excel('ExpvsPred_ANN_test.xlsx')


'''
Training and Validation loss
'''


# history.dict = history.history
# mae_values = history.dict['loss']
# val_mae_values = history.dict['val_loss']
#
# print('mae_values',mae_values)
# print('val_mae_values',val_mae_values)

# pd.DataFrame(mae_values).to_excel('training_loss.xlsx')
# pd.DataFrame(val_mae_values).to_excel('val_loss.xlsx')
#
# hist_df = pd.DataFrame(history.dict)
# hist_csv_file = '/Users/xiah/Desktop/EST/MOE_LOSS.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
# import matplotlib.pyplot as plt
#
# plt.clf()
#
# plt.plot(range(1, len(mae_values) +1 ), mae_values, 'r', label='Training loss')
# plt.plot(range(1, len(mae_values) + 1), val_mae_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Mean absolute error')
# plt.legend()
# plt.savefig('Training and validation loss.png', dpi=300)