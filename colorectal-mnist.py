############################ COLORECTAL MNIST #################################

#############################Libraries import
from keras.applications import InceptionV3
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import RMSprop,Adam
from keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
import glob
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

###########################Data Import and Data Splitting
path=r'C:\Users\Anu\Downloads\Compressed\colorectal-histology-mnist\Kather_texture_2016_image_tiles_5000\Kather_texture_2016_image_tiles_5000'
path_content=os.listdir(path)
height,width,channels=150,150,3

data,label=[],[]
for index,content in enumerate(path_content):
    image_files=glob.glob(os.path.join(path,content,'*.tif'))
    for image in image_files:
        img=cv2.imread(image)
        img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        data.append(img)
        label.append((index+1))
        
data=np.reshape(data,(len(data),height,width,channels)) #Reshape Data in (Samples,Height,Width,Channels) format


###################### Inception Model
conv_model=InceptionV3(weights='imagenet',
                 include_top=False,
                 input_shape=(height,width,channels))

#####################Feature Extraction
def extract_features(input_data):
    batch_size=10
    data_features=np.zeros(shape=(len(input_data),3,3,2048))
    data_labels=np.zeros(shape=(len(input_data)))
    for i in range(len(input_data)//batch_size):
        print('Processing Batch: %d'%(i+1))
        feature=conv_model.predict(input_data[i * batch_size : (i + 1) * batch_size])
        data_features[i * batch_size : (i + 1) * batch_size]=feature
        data_labels[i * batch_size : (i + 1) * batch_size]=label[i * batch_size : (i + 1) * batch_size]
        if i>=499:
            break
    return data_features,data_labels

data_features,data_labels=extract_features(data)

#####################Label Binarizer
lb=LabelBinarizer()
onehot_labels=lb.fit_transform(data_labels)

####################Train-Test Splitting
train_data,test_data,train_labels,test_labels=train_test_split(data_features,onehot_labels,test_size=0.2,random_state=13,shuffle=True) #Split Dataset


#####################Flatten Features(height*width*channels)
def flatten_data(input_data):
    return np.reshape(input_data,(len(input_data),input_data.shape[1]*input_data.shape[2]*input_data.shape[3]))

flatten_train_data,flatten_test_data=flatten_data(train_data),flatten_data(test_data)

#####################Learning Rate Scheduling (Just in case anyone want to apply Learning rate scheduling)
#def step_decay(epoch):
#    initial_lrate=0.1
#    drop=0.5
#    epoch_drop=5.0
#    lrate=initial_lrate*math.pow(drop,math.floor((1+epoch)/epoch_drop))
#    return lrate

#####################CallBacks Path
path=r'C:\Users\Anu\Downloads\Compressed\colorectal-histology-mnist\model_callbacks.hd5'
model_checkpoint=callbacks.ModelCheckpoint(path,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
Early_stopping=callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=1,mode='min')


#####################Creating Dense Classifier
model=Sequential()
model.add(Dense(1024,activation='relu',input_dim=(3*3*2048),use_bias=True,kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l=0.01)))
model.add(Dropout(0.4))
model.add(Dense(512,activation='relu',use_bias=True,kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(l=0.01)))
model.add(Dropout(0.4))
model.add(Dense(8,activation='sigmoid',kernel_initializer='he_normal'))

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.01,decay=0.05),metrics=['acc'])
history=model.fit(flatten_train_data,train_labels,epochs=50,batch_size=10,validation_data=(flatten_test_data,test_labels),callbacks=[model_checkpoint,Early_stopping])



##################### Metrics Plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b',color='red', label='Training acc')
plt.plot(epochs, val_acc, 'b',color='blue', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', color='red', label='Training loss')
plt.plot(epochs, val_loss, 'b',color='blue', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
###########################xxxxxxxxxxxxxxxxxxxx###############################
