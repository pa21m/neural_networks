# -*- coding: utf-8 -*-
"""appliedAI2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qTfgFlOcfG5-Cz1QtcstOGLbFs2G81qq

### **Developing a Neural Network for solving a Regression Problem**

###**Applied Artificial Intelligence**

###**COMP534**

Import Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import tensorflow as tf
'''from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.activations import relu, elu,sigmoid
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.losses import logcosh, binary_crossentropy'''


# %matplotlib inline
# %load_ext tensorboard

pip install -U keras-tuner

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import keras_tuner as kt

"""Load dataset"""

df = pd.read_csv('/content/kc_house_data.csv')
df.head()

df.info()



"""### **Cleaning Data**"""

df.isna().sum() #check for null values

df.describe()

#remove duplicate houses to keep recent house info
#house may have been sold several times so remove bias
df = df.drop_duplicates(subset="id",keep="last")
df.describe()

# replace the date column into separate columns year, month, and day
df["date"] = pd.to_datetime(df.date)
df["year"] = df.date.dt.year
df["month"] = df.date.dt.month
df["day"] = df.date.dt.day
df = df.drop("date", axis=1)

#add is_renovated column and replace all values of 0 in yr_renovated with numpy.nan
df["is_renovated"] = np.where(df.yr_renovated == 0, 0, 1)

df

df = df.drop("id", axis=1) #drop id as not needed
df

"""Find out the number of zero values in each column"""

columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15', 'year', 'month', 'day']
count = {}
for i in columns:
  count[i] = (df[i] == 0).sum()
count

"""Dropping all the rows which has 0 bathrooms"""

df.drop(df.loc[df.bathrooms == 0].index, inplace = True)

"""Correlation matrix"""

corr_matrix = df.corr().abs()
plt.figure(figsize = (20,7))
sns.heatmap(corr_matrix, annot =True, linewidth = 1)
plt.show()

"""Multicollinearity"""

corr_mul = abs(corr_matrix) >0.8
plt.figure(figsize = (20,7))
sns.heatmap(corr_mul, annot =True, linewidth = 1)
plt.show()

"""Analysing relationships between features and price"""

sns.scatterplot(data = df, x= 'price', y = 'sqft_living')
plt.show()

sns.scatterplot(data = df, x= 'price', y = 'bathrooms')
plt.show()

sns.scatterplot(data = df, x= 'price', y = 'grade')
plt.show()

sns.scatterplot(data = df, x= 'price', y = 'sqft_above')
plt.show()

sns.scatterplot(data = df, x= 'price', y = 'floors')
plt.show()

sns.scatterplot(data = df, x= 'price', y = 'bedrooms')
plt.show()

"""sqft_above has multicollinearity with other sqft_living. Since sqft_living has has higher linear relationship with price we can remove sqft_above.
Floors doesn't seem to have much of an effect on the price so we decided to remove it.
We have replaced yr_renovated with a column is_renovated since yr_renovated has a lot of 0 values



"""

df.drop(['sqft_above','floors', 'yr_renovated'],axis =1, inplace = True)

corr_matrix = df.corr().abs()
plt.figure(figsize = (20,7))
sns.heatmap(corr_matrix, annot =True, linewidth = 1)
plt.show()

sns.scatterplot(data = df, x= 'year', y = 'price')
plt.show()

sns.scatterplot(data = df, x= 'month', y = 'price')
plt.show()

"""As we can see above there is a clearly there is no relationship between the month and the year with the price of the house so we decided to remove it from the list of features"""

df_zipcode = df.groupby('zipcode').aggregate(np.mean)
df_zipcode.reset_index(inplace =True)
df_zipcode

"""**Checking if average price of a house depends on the zipcodes**

As we can see even though price and zipcode don't have a linear relationship. The prices vary a lot depending on the zipcode which make sense as price of a house will always depend on the neighbourhood. So we decided to include zipcode too despite it having a low correlation score.
"""

plt.figure(figsize = (20,7))
ax = sns.barplot(data = df_zipcode, x='zipcode', y ='price')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()

sns.boxplot( x= 'bedrooms', data=df)
plt.show()

"""Removing outliers"""

df.bedrooms.sort_values(ascending = False).head() #check for outliers

df = df[df.bedrooms <= 10] #remove outliers

"""Splitting Dataset into Test and Train"""

from sklearn.model_selection import train_test_split

x = df.drop(['price','year','month','day'],axis=1).values #features
y = df['price'].values #target

#split dataset into 80% training data and 20% test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

"""Feature Scaling"""

from sklearn.preprocessing import MinMaxScaler

#standardizing the variables to put them on the same scale
#scales all the data features in the range [0, 1]
scaler = MinMaxScaler()

scaler.fit(x_train)

#transform data so that distribution has mean 0 and s.d 1
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

"""###**Creating a Neural Network**"""

#model-building function
#hp = hyperparameter
def trial_model(act,opt):
  def model(hp):
    model = tf.keras.Sequential() #add layers in a sequence
    model.add(layers.Flatten(input_shape = (16,))) #flatten to 1D
    model.add(tf.keras.layers.Dense(units = hp.Choice('units',[2,4,10]), #no. of neurons in the layer
                                    activation=hp.Choice('activation',[act]))) #specify activation function
    model.add(tf.keras.layers.Dense(1, activation = 'linear')) #output layer
    model.compile(optimizer= hp.Choice('optimizer',[opt]), #define optimizer
                  loss = 'mean_squared_error', metrics = ['mean_squared_error']) #define loss function metrics
    return model
  return model

"""###**Testing Hyperparameters**

Model with optimizer = adam,learning rate = 0.01, activation = relu
"""

tuner1 = kt.RandomSearch(
    trial_model('relu','adam'),
    objective='val_loss', #minimize loss
    max_trials=3,
    executions_per_trial = 1,
    overwrite = True,
    directory="Trial",
    project_name="trial1", #save weights of model
)

#fitting model
tuner1.search(x_train,y_train,epochs = 6, validation_data=(x_test,y_test),callbacks=[keras.callbacks.TensorBoard("/trial/trial1/trial3_logs")])

"""***Reference*** 1"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /trial/trial1/trial3_logs #tensorboard window to check graphs for each combination of hyperparameters

"""Model with optimizer = Nadam , learning rate = 0.01, activation function = relu"""

tuner2 = kt.RandomSearch(
    trial_model('relu','nadam'),
    objective='val_loss',
    max_trials=3,
    executions_per_trial = 1,
    overwrite = True,
    directory="Trial",
    project_name="trial2",
)

tuner2.search(x_train,y_train,epochs = 6, validation_data=(x_test,y_test),callbacks=[keras.callbacks.TensorBoard("/trial/trial2/trial2_logs")])

"""***Reference*** 2"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /trial/trial2/trial2_logs #tensorboard window to check graphs for each combination of hyperparameters
#Switch to HPARAMS tab and check show metrics to check the graphs for the hyperparameters

"""model testing with optimizer = adam, learning rate =0.01, activation = elu"""

tuner3 = kt.RandomSearch(
    trial_model('elu','adam'),
    objective='val_loss',
    max_trials=3,
    executions_per_trial = 1,
    overwrite = True,
    directory="Trial",
    project_name="trial3",
)

tuner3.search(x_train,y_train,epochs = 6, validation_data=(x_test,y_test),callbacks=[keras.callbacks.TensorBoard("/trial/trial3/trial3_logs")])

"""***Reference*** 3"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /trial/trial3/trial3_logs #tensorboard window to check graphs for each combination of hyperparameters
#Switch to HPARAMS tab and check show metrics to check the graphs for the hyperparameters

"""model testing with optimizer = nadam, learning rate =0.01, activation = elu"""

tuner4 = kt.RandomSearch(
    trial_model('elu','nadam'),
    objective='val_loss',
    max_trials=3,
    executions_per_trial = 1,
    overwrite = True,
    directory="Trial",
    project_name="trial4",
)

tuner4.search(x_train,y_train,epochs = 6, validation_data=(x_test,y_test),callbacks=[keras.callbacks.TensorBoard("/trial/trial4/trial4_logs")])

"""***Reference*** 4"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /trial/trial4/trial4_logs #tensorboard window to check graphs for each combination of hyperparameters
#Switch to HPARAMS tab and check show metrics to check the graphs for the hyperparameters

"""model testing with optimizer = nadam, learning rate =[0.01,0.001,0.0001] activation = elu with fiexd 10 nodes"""

hp = kt.HyperParameters()
hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])
hp.Fixed('units',10)
tuner5 = kt.RandomSearch(
    trial_model('elu','nadam'),
    hyperparameters=hp,
    objective='val_loss',
    max_trials=3,
    executions_per_trial = 1,
    overwrite = True,
    directory="Trial",
    project_name="trial5",
)

tuner5.search(x_train,y_train,epochs = 6, validation_data=(x_test,y_test),callbacks=[keras.callbacks.TensorBoard("/trial/trial5/trial6_logs")])

"""***Reference*** 5"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /trial/trial5/trial6_logs #tensorboard window to check graphs for each combination of hyperparameters
#Switch to HPARAMS tab and check show metrics to check the graphs for the hyperparameters

"""Generalized model to find the most efficient hyperparameters"""

def model(hp):
  model = tf.keras.Sequential()
  model.add(layers.Flatten(input_shape = (16,)))
  for i in range(hp.Int('layers',2,6)):# number of hidden layers range fro 2 to 6
    model.add(tf.keras.layers.Dense(units = hp.Int('Units_' + str(i),50,100,step=10),activation=hp.Choice('act_'+str(i),['relu','elu'])))
  loss_function = hp.Choice('loss', ["mean_squared_error","log_cosh"])
  model.add(tf.keras.layers.Dense(1, activation = 'linear'))
  model.compile(optimizer=hp.Choice('optimizer',['adam','nadam'])
        ,
        loss = loss_function,
        metrics = ['mean_squared_error','mean_absolute_error'])
  return model

"""Modifying the tuner function to vary the batch size and number of epoch"""

class mytuner(kt.RandomSearch):
  def run_trial(self, trial, *args, **kwargs):
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
    kwargs['epochs'] = trial.hyperparameters.Int('epochs', 3, 6, step=3)
    return super(mytuner, self).run_trial(trial, *args, **kwargs)

hp = kt.HyperParameters()
hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4]) #the tuner randomly choose a learning rate from the list
tuner = mytuner(
    model,
    hyperparameters=hp,
    objective='val_mean_squared_error',
    overwrite = True,
    max_trials=6,
    executions_per_trial = 2,
    directory="my_dir",
    project_name="trial1",
)

tuner.search(x_train,y_train, validation_data=(x_test,y_test),callbacks=[keras.callbacks.TensorBoard("/my_dir/trial3_logs")])

"""***Reference*** 6"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /my_dir/trial3_logs #tensorboard window to check graphs for each combination of hyperparameters
#Switch to HPARAMS tab and check show metrics to check the graphs for the hyperparameters

"""### **Results**

***Reference*** 7
"""

print(tuner.results_summary()) #a summary of the combination of the top hyperparameters

best_model = tuner.get_best_models(num_models=3) #saving the top 3 models

"""1st best model"""

#added a earlystopping callback to stop the iteration when the loss function stops changing for 3 epochs
history_model1  = best_model[0].fit(x_train, y_train, epochs=100, validation_data = (x_test, y_test),callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

best_model[0].evaluate(x_test,y_test)

#ploting the model loss
plt.plot(history_model1.history['mean_squared_error'])
plt.plot(history_model1.history['val_mean_squared_error'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_test_model1 = best_model[0].predict(x_test)

#plotting True values vs Predicted Values
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test_model1,y_test)
plt.plot(y_test,y_test,'r')
plt.title('True values vs Predicted Values')
plt.ylabel('True values')
plt.xlabel('Predicted values')

"""2nd Best model"""

history_model2  = best_model[1].fit(x_train, y_train, epochs=100, validation_data = (x_test, y_test),callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

best_model[1].evaluate(x_test,y_test)

#ploting the model loss
plt.plot(history_model2.history['loss'])
plt.plot(history_model2.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_test_model2 = best_model[1].predict(x_test)

#plotting True values vs Predicted Values
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test_model2,y_test)
plt.plot(y_test,y_test,'r')
plt.title('True values vs Predicted Values')
plt.ylabel('True values')
plt.xlabel('Predicted values')

"""3rd Best model"""

history_model3  = best_model[2].fit(x_train, y_train, epochs=100, validation_data = (x_test, y_test),callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

best_model[2].evaluate(x_test,y_test)

#ploting the model loss
plt.plot(history_model3.history['loss'])
plt.plot(history_model3.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_test_model3 = best_model[2].predict(x_test)

#plotting True values vs Predicted Values
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test_model3,y_test)
plt.plot(y_test,y_test,'r')
plt.title('True values vs Predicted Values')
plt.ylabel('True values')
plt.xlabel('Predicted values')