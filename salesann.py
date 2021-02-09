import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel('Superstore.xls')

df['Order Date'] = pd.to_datetime(df['Order Date']).dt.date
df['Ship Date'] = pd.to_datetime(df['Ship Date']).dt.date
df.loc[df['Ship Date'].notnull(), 'Days'] = df['Ship Date'] - df['Order Date']

df['Days']=df.apply(lambda row: row.Days.days, axis=1)


arr = df.to_numpy()



X=arr[:,[4,7,12,14,15,18,19,20,21]]
y=arr[:,17]

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_X_3 = LabelEncoder()
X[:, 2] = labelencoder_X_3.fit_transform(X[:, 2])
labelencoder_X_4 = LabelEncoder()
X[:, 3] = labelencoder_X_4.fit_transform(X[:, 3])
labelencoder_X_5 = LabelEncoder()
X[:, 4] = labelencoder_X_5.fit_transform(X[:, 4])
transformer = ColumnTransformer(transformers=[("OneHot",OneHotEncoder(),[0,4])],remainder="passthrough")
X = transformer.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_testandval, y_train, y_testandval = train_test_split(X, y, test_size = 0.3, random_state = 0)
X_val, X_test, Y_val, y_test = train_test_split(X_testandval, y_testandval, test_size=0.5)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras#Keras is an open-source neural-network library written in Python.
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()#The model type is Sequential. It allows to build a model layer by layer. Each layer has weights that correspond to the layer the follows it.
#Dense is the layer type.In a dense layer, all nodes in the previous layer connect to the nodes in the current layer

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, activation='relu', input_shape=(28,)))
#add() function is used to add layers to model.Here we have built a layer with 6 neurons.The activation function used is ReLU or Rectified Linear Activation. The input shape specifies the number of columns in the input

# Adding the second hidden layer
classifier.add(Dense(6, activation = 'relu'))

# Adding the output layer
classifier.add(Dense(1,activation = 'relu'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
#mse loss function and adam optimiser binary_crossentropy
#The optimizer controls the learning rate.Here adam is used as optmizer. The adam optimizer adjusts the learning rate throughout training.Loss function is binary crossentropy

# Fitting the ANN to the Training set
hist=classifier.fit(X_train, y_train, batch_size = 6, epochs = 100,validation_split=0.25)#validation_data=(X_val, Y_val)
#The number of epochs is the number of times the model will cycle through the data. 

classifier.evaluate(X_test,y_test)[1]

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)

#Part 4 Visualisation
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

plt.plot(y_pred)
plt.plot(y_test)
plt.title('Sales')
plt.ylabel('Sales')
plt.xlabel('Date')
plt.legend(['Predicted', 'Actual'], loc='lower right')
plt.show()




from keras.layers import Dropout
from keras import regularizers

