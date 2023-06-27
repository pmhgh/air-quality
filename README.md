# air-quality
We want to use the layers of Keras to process our data of air quality 
In this GitHub repository, our focus is on preparing the data for analysis. We begin by normalizing the data to ensure consistency and eliminate any variations in scale. This step is crucial for enhancing the performance of our machine learning models.

After data normalization, we proceed to address the presence of outliers. Outliers can adversely affect the accuracy and reliability of our analysis, so we implement techniques to identify and remove them. By eliminating outliers, we aim to create a more robust and representative dataset for training our models.

Once the data is properly prepared, we move on to the core of our work—building and training our machine learning model. We leverage neural networks as the foundation for our model and fine-tune it using appropriate loss functions to minimize errors and improve accuracy. By optimizing these parameters, we aim to achieve the best possible performance from our neural network model.

To assess the effectiveness of our neural network, we compare its performance with other machine learning algorithms. This comparative analysis allows us to evaluate the strengths and weaknesses of different approaches and determine which algorithm yields the most accurate predictions for our specific dataset.

Through this repository, we aim to provide a comprehensive framework for data preparation, model training, and performance evaluation in machine learning. By following these steps, researchers and practitioners can gain insights into the effectiveness of various algorithms and make informed decisions when working with their own datasets.
! pip install -q kaggle

import pandas as pd
df = pd.read_csv('AQI.csv')

df

print(len(df) , df.shape)

print(df.info())

df.drop(columns = ['Id' ,'Mounths'], inplace=True, axis=1)

print(len(df) , df.shape , df.info())

df.head(5)

df.isnull().sum()

import seaborn as sns
sns.displot(df['PM10 in æg/m3'])

sns.displot(df['SO2 in æg/m3'])

sns.distplot(df['NOx  in æg/m3'])

sns.displot(df.AQI)

df['PM10 in æg/m3'].fillna(df['PM10 in æg/m3'].mean(),inplace = True)
df['SO2 in æg/m3'].fillna(df['SO2 in æg/m3'].mean(),inplace = True)
df['NOx  in æg/m3'].fillna(df['NOx  in æg/m3'].mean(),inplace = True)
df['AQI'].fillna(df['AQI'].mean(),inplace = True)

df.isnull().sum()

sns.boxplot(df.AQI)

q1 = df.AQI.quantile(.25)
q3 = df.AQI.quantile(.75)
IQR = q3 -q1
IQR

upper_limit = q3 + 1.5 * IQR
lower_limit = q1 - 1.5 * IQR

df.median()

# we insert 104(median) for data which are more than upper limit.then , we insert the rest of data for less than upper limit.

import numpy as np
df['AQI'] = np.where(df['AQI'] > upper_limit , 104 , df['AQI'])

sns.boxplot(df.AQI)

x = df.drop(columns = ['AQI'] , axis = 1)
x.head()

y = df.AQI
y.head()

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size = .3 , random_state = 42)

xtrain.shape , xtest.shape , ytrain.shape , ytest.shape

x_val = xtrain[:15]
partial_x_train = xtrain[15:]
y_val = ytrain[:15]
partial_y_train = ytrain[15:]

"""# Go Ahead"""

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(xtrain , ytrain)

y_pred = lm.predict(xtest)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest , y_pred)
print('Mean Squared Error:' , mse)

import numpy as np
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest , y_pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:' , rmse)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(ytest , y_pred)
print('Mean Absolute Error' , mae)

"""# Neural Network"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

reg_model = Sequential()
reg_model.add(Dense(7 , activation = 'relu'))
reg_model.add(Dense(256, activation = 'relu'))
reg_model.add(Dense(64, activation = 'relu'))
reg_model.add(Dense(32, activation = 'relu'))
reg_model.add(Dense(16, activation = 'relu'))
reg_model.add(Dense(1, activation = 'softmax'))


reg_model.compile(optimizer="adam",loss="mse",metrics=['mse','mae'])
reg_model.fit(xtrain,ytrain,epochs=10,batch_size=2,validation_data=(xtest,ytest))

y_pred = reg_model.predict(xtest)

from tensorflow import keras
#MEAN SQUARD ERROR ,ROOT MEAN SQUARED ERROR ,MEAN ABSOLUTE ERROR
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest, y_pred)
print('Mean Squared Error:', mse)

import numpy as np
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest, y_pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(ytest, y_pred)
print('Mean Absolute Error:', mae)

corr_matrix = df.corr()
plt.figure(figsize = ( 10 , 8))
sns.heatmap(corr_matrix , annot = True , cmap = 'coolwarm' , linewidths = .5)
plt.title('Correlation Matrix')
plt.show()

"""# Random Forest"""

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain , ytrain)
y_pred = rf.predict(xtest)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest , y_pred)
print('Mean Squared Error:' , mse)

import numpy as np
from sklearn.metrics import mean_squared_error
mse_2 = mean_squared_error(ytest , y_pred)
rmse_2 = np.sqrt(mse_2)
print('Root Mean Squared Error:' , rmse)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(ytest , y_pred)
print('Mean Absolute Error:' , mae)

"""# Decision Tree"""

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(xtrain , ytrain)
y_pred = dt.predict(xtest)
from sklearn.metrics import mean_squared_error
mse_3 = mean_squared_error(ytest, y_pred)
print('Mean Squared Error:', mse)
#ROOT MEAN SQUARED ERROR
import numpy as np
from sklearn.metrics import mean_squared_error
mse_3= mean_squared_error(ytest, y_pred)
rmse_3 = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)
#MEAN ABSOLUTE ERROR
from sklearn.metrics import mean_absolute_error
mae_3 = mean_absolute_error(ytest, y_pred)
print('Mean Absolute Error:', mae)
