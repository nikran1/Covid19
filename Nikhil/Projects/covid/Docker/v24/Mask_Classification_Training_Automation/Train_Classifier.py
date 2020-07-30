
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras import backend as K
from keras.regularizers import l2
from keras import layers
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import SeparableConv2D

import keraTomodeloptimizer

train_data_dir = 'train'
validation_data_dir = 'val'
test_data_dir = 'test'


# input_shape = (64, 64, 3)
input_shape = (64, 64, 1)

if(input_shape[2] == 1):
    Color_Mode = "grayscale"
else:
    Color_Mode = 'rgb'

batch_size = 20

# img_width, img_height = 64, 64
img_width = input_shape[0]
img_height = input_shape[1]



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)



generator_train = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), color_mode= Color_Mode, batch_size=batch_size )
generator_val = val_datagen.flow_from_directory( validation_data_dir, target_size=(img_width, img_height), color_mode= Color_Mode,  batch_size=batch_size, shuffle = False)
generator_test = test_datagen.flow_from_directory(test_data_dir, target_size=(img_width, img_height), color_mode= Color_Mode, batch_size=batch_size, shuffle = False)

batch_size = 128
num_classes = 3

regularization = l2(0.01)
img_input = Input(input_shape)

x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                      use_bias=False)(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                      use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)


residual = Conv2D(16, (1, 1), strides=(2, 2),
                padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(16, (3, 3), padding='same',
                  kernel_regularizer=regularization,
                  use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(16, (3, 3), padding='same',
                  kernel_regularizer=regularization,
                  use_bias=False)(x)
x = BatchNormalization()(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])


residual = Conv2D(32, (1, 1), strides=(2, 2),
                padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(32, (3, 3), padding='same',
                  kernel_regularizer=regularization,
                  use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(32, (3, 3), padding='same',
                  kernel_regularizer=regularization,
                  use_bias=False)(x)
x = BatchNormalization()(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])


residual = Conv2D(64, (1, 1), strides=(2, 2),
                padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(64, (3, 3), padding='same',
                  kernel_regularizer=regularization,
                  use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(64, (3, 3), padding='same',
                  kernel_regularizer=regularization,
                  use_bias=False)(x)
x = BatchNormalization()(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])


residual = Conv2D(128, (1, 1), strides=(2, 2),
                padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(128, (3, 3), padding='same',
                  kernel_regularizer=regularization,
                  use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(128, (3, 3), padding='same',
                  kernel_regularizer=regularization,
                  use_bias=False)(x)
x = BatchNormalization()(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

x = Conv2D(num_classes, (3, 3), padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax',name='predictions')(x)

model = Model(img_input, output)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

nb_train_samples = 2000
nb_validation_samples = 800
# batch_size = 16
batch_size = 100
epochs = 1
fp = "FP16"

val_loss = [None] * epochs

model_array = []

for i in range(epochs):
    print(i,"\n\n\n")
    out2 = model.fit_generator(
        generator_train,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=1,
        validation_data=generator_val,
        validation_steps=nb_validation_samples // batch_size, workers = 1)

    val_loss[i] = out2.history['val_loss']
    model_array.append(model)

best_model_index = val_loss.index(min(val_loss))

print(best_model_index, val_loss[best_model_index])

model_array[best_model_index].save('model/mask_recognition_v4_updated.hdf5')


# keraTomodeloptimizer.keraTomodeloptimizer('1',1,input_shape[0],input_shape[1],input_shape[2])
keraTomodeloptimizer.keraTomodeloptimizer('1',input_shape,fp)
