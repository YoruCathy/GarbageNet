from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras_preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
from keras.models import Sequential
from keras import optimizers, losses
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import pickle
import os
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class GarbageClassification():
    """
    Class for GarbageNet
    Final Project for EE369,2020 Spring
    By Cathy
    """
    def __init__(self, gpu="1", data_dir='./data', save_dir= "./result/",size=224, backbone="VGG16", infer = False, num_classes=6,logname=None):
        if len(gpu)!=1:
            raise NotImplementedError("Multi GPU Training not supported.")
        self.gpu = gpu

        if not os.path.exists(data_dir):
            raise NameError("Data path not found.")
        if not os.path.exists(save_dir):
            raise NameError("Path for model saving not found.")
        self.data_dir = data_dir
        self.save_dir = save_dir+"{epoch:02d}_{val_accuracy:.2f}_"+backbone+".h5"
        self.size = size
        self.backbone = backbone
        self.use_pretrained_weights = True
        self.bs = 32
        self.num_epoch = 100
        self.infer = infer
        self.num_classes = num_classes
        self.CLASSES = ["cardboard","glass","metal","paper","plastic","trash"]
        self.logname=logname
    def set_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

    def prepare_pipeline(self):
        """
        This function defines the train and val pipeline.
        In train pipeline, we define data auguments.
        """
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
        """
        This fuction trains the model.
        """
        print("Traing "+self.backbone)
        if self.backbone == "inception":
            if self.use_pretrained_weights:
                backbone = InceptionV3(weights = "imagenet", include_top=False, input_shape=(self.size,self.size, 3))
                backbone.trainable = False
            else:
                backbone = InceptionV3(include_top=False, input_shape=(self.size,self.size, 3))
                backbone.trainable = True

        elif self.backbone == "VGG16":
            if self.use_pretrained_weights:
                backbone = VGG16(weights = "imagenet", include_top=False, input_shape=(self.size,self.size, 3))
                backbone.trainable = False
            else:
                backbone = VGG16(include_top=False, input_shape=(self.size,self.size, 3))
                backbone.trainable = True

        elif self.backbone == "MobileNet":
            if self.use_pretrained_weights:
                backbone = MobileNet(weights = "imagenet", include_top=False, input_shape=(self.size,self.size, 3))
                backbone.trainable = False
            else:
                backbone = MobileNet(include_top=False, input_shape=(self.size,self.size, 3))
                backbone.trainable = True

        else:
            raise NotImplementedError("Do not support this kind of backbone.")

        model = Sequential([
                backbone,
                GlobalAveragePooling2D(),
                Dropout(0.1),
                Dense(1024, activation='relu'),
                Dense(6, activation='softmax')
            ])

        """
        This part can be modified to change learning rate schedule.
        We will make it cleaner in the future. these messy code is because of the coming ddl.
        """
        initial_learning_rate = 0.001
        lr_schedule = keras.experimental.CosineDecay(
            initial_learning_rate, decay_steps=1000, alpha=0.0, name=None
        )

        opt = keras.optimizers.RMSprop(learning_rate=lr_schedule)
        # opt=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
        
        checkpoint = ModelCheckpoint(self.save_dir, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        steps_per_epoch = pipeline["train_pipeline"].n // self.bs
        validation_steps = pipeline["val_pipeline"].n // self.bs
        history = model.fit_generator(generator=pipeline["train_pipeline"], epochs=self.num_epoch, steps_per_epoch=steps_per_epoch,
                            validation_data=pipeline["train_pipeline"], validation_steps=validation_steps,
                            callbacks=callbacks_list)
        # Here the log is saved, and the model is also saved.
        with open(self.backbone+self.logname+'train_log.txt', 'wb') as log:
            pickle.dump(history.history, log) 
        log.close()


    def single_test(self, ckpt_path='./result/model_1953-0.93.h5', test_path="",test_name = ""):
        """
        To test and visualize a single image.
        test_path is the file path, test name is the image name 
        """
        from keras.models import load_model
        from keras.preprocessing import image
        from keras.applications.inception_v3 import preprocess_input
        model = load_model(ckpt_path)
        # model.summary()
        x = image.load_img(test_path+test_name,target_size=(self.size,self.size))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        pred_index,confidence = self.decode_predictions_garbage_classification(preds)
        self.visualize_result(pred_index,confidence,test_path=test_path,test_name=test_name)
    
    def batch_test(self, ckpt_path='result/64_0.86_MobileNet.h5',batch_test_path = "./unseen/"):
        """
        To test images in a file.
        and save the visualized images
        """
        img_list = os.listdir(batch_test_path)
        for img in tqdm(img_list):
            self.single_test(ckpt_path = ckpt_path, test_path=batch_test_path,test_name=img)

    def decode_predictions_garbage_classification(self, preds):
        """
        Decode the results from the network language to human language.
        """
        pred_result = preds[0]
        pred_index = np.argmax(pred_result)
        confidence=pred_result[pred_index]
        return pred_index,confidence

    def visualize_result(self,pred_index,confidence,save_pic=True,test_path="",test_name = ""):
        """
        As the name suggests
        """
        import cv2 as cv
        img = cv.imread(test_path+test_name)
        text=self.CLASSES[pred_index]+" "+str(confidence)
        cv.putText(img,text,(40,50),cv.FONT_HERSHEY_PLAIN,2.0,(100,100,255),2)
        if save_pic:
            cv.imwrite("./vis_trash/"+test_name,img)
        else:
            cv.imshow(img)
        
    def vis_log(self,log_path):
        """
        As the name suggests
        """
        path=open(log_path,'rb')
        log = pickle.load(path,encoding='utf-8')
        for item in log:
            x = [i for i in range(len(log[item]))]
            y = log[item]
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set(xlabel='epoch', ylabel=str(item),
                title='Visualization of '+str(item)+' log')
            ax.grid()
            fig.savefig("./log_vis/"+str(item)+".png")
    
    def get_flops(self,model_h5_path):
        """
        As the name suggests
        """
        import tensorflow as tf
        session = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        with graph.as_default():
            with session.as_default():
                model = tf.keras.models.load_model(model_h5_path)

                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(graph=graph,
                                                    run_meta=run_meta, cmd='op', options=opts)

                return flops.total_float_ops

    def test_speed(self, model_h5_path,batch_test_path = "./data/cardboard/"):
        """
        As the name suggests.
        In the form of FPS.
        """
        from keras.models import load_model
        from keras.preprocessing import image
        from keras.applications.inception_v3 import preprocess_input
        import time
        model = load_model(model_h5_path)
        img_list = os.listdir(batch_test_path)
        deltas=0
        for img in tqdm(img_list):
            x = image.load_img(batch_test_path+img,target_size=(self.size,self.size))
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            start = time.time()
            preds = model.predict(x)
            end=time.time()
            delta=end-start
            deltas+=delta
        speed=deltas/len(img_list)
        print(60/speed)

    
