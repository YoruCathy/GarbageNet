from keras.applications.inception_v3 import InceptionV3
from keras_preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
from keras.models import Sequential
from keras import optimizers, losses
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import pickle
import os
from tqdm import tqdm

class GarbageClassification():
    def __init__(self, gpu="1", data_dir='./data', save_dir= "./result/",size=256, backbone="inception"):
        if len(gpu)!=1:
            raise NotImplementedError("Multi GPU Training not supported.")
        self.gpu = gpu

        if not os.path.exists(data_dir):
            raise NameError("Data path not found.")
        if not os.path.exists(save_dir):
            raise NameError("Path for model saving not found.")
        self.data_dir = data_dir
        self.save_dir = save_dir+"model_{epoch:02d}_{val_accuracy:.2f}"+backbone+".h5"
        self.size = size
        self.backbone = backbone
        self.use_pretrained_weights = True
        self.bs = 32
        self.num_epoch = 1000
    
    def set_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

    def prepare_pipeline(self):
        pipeline = {
            "train_pipeline": ImageDataGenerator(
                horizontal_flip=True,
                vertical_flip=True,
                rescale=1. / 255,
                validation_split=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=30,
            ).flow_from_directory(
                directory=self.data_dir,
                target_size=(self.size,self.size),
                subset='training',
            ),

            "val_pipeline": ImageDataGenerator(
                rescale=1 / 255,
                validation_split=0.1,
            ).flow_from_directory(
                directory=self.data_dir,
                target_size=(self.size,self.size),
                subset='validation',
            ),
        }
        return pipeline

    def train(self,pipeline):
        if self.backbone == "inception":
            if self.use_pretrained_weights:
                backbone = InceptionV3(weights = "imagenet", include_top=False, input_shape=(self.size,self.size, 3))
                backbone.trainable = False
            else:
                backbone = InceptionV3(include_top=False, input_shape=(self.size,self.size, 3))
                backbone.trainable = True

            model = Sequential([
                backbone,
                GlobalAveragePooling2D(),
                Dropout(0.1),
                Dense(1024, activation='relu'),
                Dense(6, activation='softmax')
            ])
            model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
            
            checkpoint = ModelCheckpoint(self.save_dir, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]

            steps_per_epoch = pipeline["train_pipeline"].n // self.bs
            validation_steps = pipeline["val_pipeline"].n // self.bs
            history = model.fit_generator(generator=pipeline["train_pipeline"], epochs=self.num_epoch, steps_per_epoch=steps_per_epoch,
                              validation_data=pipeline["train_pipeline"], validation_steps=validation_steps,
                              callbacks=callbacks_list)
            with open(self.backbone+'train_log.txt', 'wb') as log:
                pickle.dump(history.history, log) 
            log.close()

        else:
            raise NotImplementedError("Do not support this kind of backbone.")

                     
