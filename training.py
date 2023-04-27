from keras.utils import to_categorical, plot_model 
from keras.optimizers import SGD, Adam
from keras import backend
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from model import arquitetura
import datetime
import h5py
import time

EPOCHS = 30
CLASS = 21
FILE_NAME = 'cnn_model_LIBRAS_v2'

def getDateStr():
        return str('{date:%Y%m%d_%H%M}').format(date=datetime.datetime.now())

def getTimeMin(start, end):
        return (end - start)/60

print('[INFO] Carregando o dataset com keras.preprocessing.image.ImageDataGenerator')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True, 
        validation_split=0.25)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

train_data = train_datagen.flow_from_directory(
        '/dataset/train',
        target_size=(64, 64),
        color_mode = 'rgb',
        batch_size=32,
        shuffle=False,
        class_mode='categorical')

test_data = test_datagen.flow_from_directory(
        '/dataset/test',
        target_size=(64, 64),
        color_mode = 'rgb',
        batch_size=32,
        shuffle=False,
        class_mode='categorical')

# inicializar e otimizar modelo
print("[INFO] Inicializando e otimizando a CNN...")

inicio = time.time()
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

model = arquitetura.build(64, 64, 3, CLASS)
model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
              metrics=["accuracy"])

# treinando a CNN

print("[INFO] Treinando a CNN...")
classifier = model.fit_generator(
        train_data,
        steps_per_epoch=(train_data.n // train_data.batch_size),
        epochs=EPOCHS,
        validation_data = test_data,
        validation_steps= (test_data.n // test_data.batch_size),
        shuffle = False,
        callbacks = [early_stopping]
      )

print("[INFO] Salvando modelo treinado ...")

#para todos arquivos ficarem com a mesma data e hora. Armazeno na variavel
file_date = getDateStr()
model.save('../models/'+FILE_NAME+file_date+'.h5')
print('[INFO] modelo: ../models/'+FILE_NAME+file_date+'.h5 salvo!')

print('[INFO] Summary: ')
model.summary()

# Avaliando a CNN

print("\n[INFO] Avaliando a CNN...")
score = model.evaluate_generator(generator=test_data, steps=(test_data.n // test_data.batch_size), verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))