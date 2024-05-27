# spell-checker: disable
import tensorflow as tf
import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  MaxPooling2D, Flatten, Dense, Dropout,Convolution2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import pickle
import time
# 数据处理
datagen = ImageDataGenerator(
    validation_split=0.2
)

TrainingImagePath = '/Users/lkh/Desktop/real time face detection/myFace'
# 从目录中读取数据
train_set = datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_set = datagen.flow_from_directory(
    TrainingImagePath,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print(validation_set.class_indices)

# creating lookup table for class indices
TrainClasses = train_set.class_indices
ResultMap = {}  
for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
    ResultMap[faceValue] = faceName


with open('face_classes.pkl', 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

print("mapping of face and its id:", ResultMap)

OutputNeurons = len(ResultMap)
print("the number of output neurons:", OutputNeurons)

# 构建模型
classifier = Sequential()

''' step 1 - Convolution
# adding the first layer of CNN
# we are using the format (64, 64, 3) because we are using tensorflow backend
# it means 64x64 pixels and 3 channels for RGB
'''
classifier.add(Convolution2D(32, (3, 3) ,input_shape=(64, 64, 3), activation='relu'))

'''step 2 max pooling'''
classifier.add(MaxPooling2D(pool_size=(2, 2)))

''' additional layer of convolution for better accuracy'''
classifier.add(Convolution2D(64, (5, 5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

'''step 3 flattening'''
classifier.add(Flatten())

'''step 4 full connection'''
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(OutputNeurons, activation='softmax'))

'''compiling the CNN'''
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
StartTime = time.time()


history = classifier.fit_generator(
    train_set,
    steps_per_epoch=10,
    epochs=8,
    validation_data=validation_set,
    validation_steps=10,
    )

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



EndTime = time.time()
print("###### TOTAL TIME TAKEN: ", EndTime - StartTime, " ######")

# 保存模型
classifier.save('face_model.h5')
print("model saved successfully")