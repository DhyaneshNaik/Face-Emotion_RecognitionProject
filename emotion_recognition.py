import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,BatchNormalization
#from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv("data/fer2013.csv")
#print(data.info())
#print(data.head())
#print(data['Usage'].value_counts())
X_train,X_test, y_train, y_test = [],[],[],[]

for index, row in data.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val,'float32'))
            y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val,'float32'))
            y_test.append(row['emotion'])
    except:
        print(f"Error occurred at index{index} and row: {row}")

#X_train = scale(np.array(X_train,'float32'),axis=0,with_mean=True,with_std=True)
#X_test = scale(np.array(X_test,'float32'),axis=0,with_mean=True,with_std=True)
#OR
X_train = np.array(X_train,'float32') / 255
X_test = np.array(X_test,'float32') / 255
y_train = np.array(y_train,dtype='uint8')
y_test = np.array(y_test,dtype='uint8')
y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)
#X_train -= np.mean(X_train,axis=0)
#X_train /= np.std(X_train,axis=0)
#X_test -= np.mean(X_test,axis=0)
#X_test /= np.std(X_test,axis=0)

#print(f"X_train sample data: {X_train[:2]}")
#print(f"X_test sample data: {X_test[:2]}")
#print(f"y_train sample data: {y_train[:2]}")
#print(f"y_test sample data: {y_test[:2]}")


num_labels = 7
batch_size = 64
epochs = 100
width,height = 48,48
print("Xtrain: ",X_train.shape)
print("Xtest:",X_test.shape)
X_train = X_train.reshape(X_train.shape[0],width,height,1)
#X_train = X_train.reshape(width,height,1)
X_test = X_test.reshape(X_test.shape[0],width,height,1)
#X_test = X_test.reshape(width,height,1)
print("Xtrain: ",X_train.shape)
print("Xtest:",X_test.shape)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',input_shape=X_train.shape[1:], kernel_regularizer=l2(0.01)))#(X_train.shape[1:])
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same",activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same",activation='relu'))
model.add(BatchNormalization())
#model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same",activation='relu'))
#model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same",activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
#model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same",activation='relu'))
#model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same",activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same",activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same",activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same",activation='relu'))
model.add(BatchNormalization())
#model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same",activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_labels,activation='softmax'))
print(model.summary())
adam = Adam(lr = 0.001)
model.compile(loss="categorical_crossentropy",optimizer=adam,metrics=['accuracy'])

#early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=6, mode='auto')

history = model.fit(X_train,y_train,epochs=epochs,verbose=2,validation_data=(X_test,y_test),batch_size=batch_size,shuffle=True)#,callbacks=[early_stopper]

fer_json = model.to_json()
with open('fer.json','w') as f:
    f.write(fer_json)
model.save_weights('fer_weights.h5')

plt.plot(history.history['val_accuracy'],label="Validation Acc")
plt.title("Validation Accuracy")
plt.legend()
fig11 = plt.gcf()
plt.show()
plt.draw()
fig11.savefig("Validation Accuracy",dpi=100)

plt.plot(history.history['val_accuracy'],label="Validation Accuracy")
plt.plot(history.history['accuracy'],label="Training Accuracy")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Accuracy")
plt.legend()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig("Accuracy",dpi=100)

plt.plot(history.history['val_loss'],label="Validation Loss")
plt.plot(history.history['loss'],label="Training Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Loss")
plt.legend()
fig2 = plt.gcf()
plt.show()
plt.draw()
fig2.savefig("Loss",dpi=100)