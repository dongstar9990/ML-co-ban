from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier ,MLPRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
# sai so trung binh va sai so tuyet doi
from sklearn.metrics import mean_squared_error,mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier ,MLPRegressor
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.engine.training import optimizers
from keras.engine.sequential import Sequential
from keras.optimizers import Adam

Data={'x':[13,23,13,14,43,25,55,33,1,44,42,12,41,53,24],
      'y':[12,45,21,41,53,13,12,44,1,45,12,13,55,33,42]}
df=DataFrame(Data, columns=['x','y'])
df.head()

# chia lam 3 cluster
kmeans=KMeans(n_clusters=3).fit(df)
centroids=kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'],df['y'], c=kmeans.labels_.astype(float),s=50,alpha=0.5)
plt.scatter(centroids[:,0],centroids[:,1], c="red", s=50)

kmeans=KMeans(n_clusters=4).fit(df)

centroidsv2=kmeans.cluster_centers_

print(centroidsv2)

plt.scatter(df['x'],df['y'], c=kmeans.labels_.astype(float), s=50,alpha=0.5)
plt.scatter(centroidsv2[:,0],centroidsv2[:,1], c="blue" ,s=50 , alpha=1)
plt.show()

# day la data quyet dinh
RealEstate=pd.read_csv('RealEstate (2).csv')
# plt.show()
X=RealEstate.iloc[:,2:7]
X

y=RealEstate['Y house price of unit area']

y
X_train, X_test ,y_train,y_test=train_test_split(X,y,random_state=1)
y_train

NN=MLPRegressor(max_iter=300,activation="relu", hidden_layer_sizes=(100,100))
NN.fit(X_train,y_train)
# sau khi train xong du doan ket qua
NN_pred=NN.predict(X_test)
# trung binh binh phuong sai so
MSE=mean_squared_error(y_test,NN_pred)
print(MSE)

MAE=mean_absolute_error(y_test,NN_pred)
print(MAE)


model=Sequential()
model.add(Dense(100,input_dim=X_train.shape[1] , activation='relu'))
model.add(Dense(1))
# train

model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mae'])
history=model.fit(X_train,y_train,epochs=200)

y_pred=model.predict(X_test)
y_pred[1:10]

MSE1=mean_squared_error(y_test,y_pred)
print(MSE1)
MAE1=mean_absolute_error(y_test,y_pred)
print(MAE1)
model.evaluate(X_test,y_test)
plt.plot(history.history['loss'])