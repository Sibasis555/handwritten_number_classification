import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from hwn_cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import tensorflow as tf
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    def update_base_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=self.config.params_image_size))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.config.params_classes, activation='softmax'))
        self.model.summary()
        self.model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
        self.save_model(path=self.config.updated_base_model_path, model=self.model)
        return self.model
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)