"""Handwritten Text Recognition Neural Network"""


from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, add,Softmax, AveragePooling2D, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model


import os
import logging

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    logging.disable(logging.WARNING)
except AttributeError:
    pass

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
from contextlib import redirect_stdout
from tensorflow.keras import backend as K
from tensorflow.keras import Model

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm

from network.layers import FullGatedConv2D, GatedConv2D, OctConv2D
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape
from network.cbam import cbam_block 
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.layers import Conv1D, SpatialDropout1D
from tensorflow.keras.layers import multiply
from tensorflow.keras.activations import selu


'''
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
'''
"""
HTRModel Class based on:
    Y. Soullard, C. Ruffino and T. Paquet,
    CTCModel: A Connectionnist Temporal Classification implementation for Keras.
    ee: https://arxiv.org/abs/1901.07957, 2019.
    github: https://github.com/ysoullard/HTRModel


The HTRModel class use Tensorflow 2 Keras module for the use of the
Connectionist Temporal Classification (CTC) with the Hadwritten Text Recognition (HTR).

In a Tensorflow Keras Model, x is the input features and y the labels.
"""


class HTRModel:

    def __init__(self,
                 architecture,
                 input_size,
                 vocab_size,
                 greedy=False,
                 beam_width=10,
                 top_paths=1,
                 stop_tolerance=20,
                 reduce_tolerance=15,
                 cooldown=0):
        """
        Initialization of a HTR Model.

        :param
            architecture: option of the architecture model to build and compile
            greedy, beam_width, top_paths: Parameters of the CTC decoding
            (see ctc decoding tensorflow for more details)
        """

        self.architecture = globals()[architecture]
        self.input_size = input_size
        self.vocab_size = vocab_size

        self.model = None
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = max(1, top_paths)

        self.stop_tolerance = stop_tolerance
        self.reduce_tolerance = reduce_tolerance
        self.cooldown = cooldown

    def summary(self, output=None, target=None):
        """Show/Save model structure (summary)"""

        self.model.summary()

        if target is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, target), "w") as f:
                with redirect_stdout(f):
                    self.model.summary()

    def load_checkpoint(self, target):
        """ Load a model with checkpoint file"""

        if os.path.isfile(target):
            if self.model is None:
                self.compile()

            self.model.load_weights(target)

    def get_callbacks(self, logdir, checkpoint, monitor="val_loss", verbose=0):
        """Setup the list of callbacks for the model"""

        callbacks = [
            CSVLogger(
                filename=os.path.join(logdir, "epochs.log"),
                separator=";",
                append=True),
            TensorBoard(
                log_dir=logdir,
                histogram_freq=10,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=self.stop_tolerance,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=self.reduce_tolerance,
                cooldown=self.cooldown,
                verbose=verbose)
        ]

        return callbacks

    def compile(self, learning_rate=None, initial_step=0):
        """
        Configures the HTR Model for training/predict.

        :param optimizer: optimizer for training
        """

        # define inputs, outputs and optimizer of the chosen architecture
        inputs, outputs = self.architecture(self.input_size, self.vocab_size + 1)

        if learning_rate is None:
            learning_rate = CustomSchedule(d_model=self.vocab_size + 1, initial_step=initial_step)
            self.learning_schedule = True
        else:
            self.learning_schedule = False

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        # create and compile
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer, loss=self.ctc_loss_lambda_func)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):
        """
        Model training on data yielded (fit function has support to generator).
        A fit() abstration function of TensorFlow 2.

        Provide x parameter of the form: yielding (x, y, sample_weight).

        :param: See tensorflow.keras.Model.fit()
        :return: A history object
        """

        # remove ReduceLROnPlateau (if exist) when use schedule learning rate
        if callbacks and self.learning_schedule:
            callbacks = [x for x in callbacks if not isinstance(x, ReduceLROnPlateau)]

        out = self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                             callbacks=callbacks, validation_split=validation_split,
                             validation_data=validation_data, shuffle=shuffle,
                             class_weight=class_weight, sample_weight=sample_weight,
                             initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps, validation_freq=validation_freq,
                             max_queue_size=max_queue_size, workers=workers,
                             use_multiprocessing=use_multiprocessing, **kwargs)
        return out

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=1,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                ctc_decode=True):
        """
        Model predicting on data yielded (predict function has support to generator).
        A predict() abstration function of TensorFlow 2.

        Provide x parameter of the form: yielding [x].

        :param: See tensorflow.keras.Model.predict()
        :return: raw data on `ctc_decode=False` or CTC decode on `ctc_decode=True` (both with probabilities)
        """

        if verbose == 1:
            print("Model Predict")

        out = self.model.predict(x=x, batch_size=batch_size, verbose=verbose, steps=steps,
                                 callbacks=callbacks, max_queue_size=max_queue_size,
                                 workers=workers, use_multiprocessing=use_multiprocessing)

        if not ctc_decode:
            return np.log(out.clip(min=1e-8)), []

        steps_done = 0
        if verbose == 1:
            print("CTC Decode")
            progbar = tf.keras.utils.Progbar(target=steps)

        batch_size = int(np.ceil(len(out) / steps))
        input_length = len(max(out, key=len))

        predicts, probabilities = [], []

        while steps_done < steps:
            index = steps_done * batch_size
            until = index + batch_size

            x_test = np.asarray(out[index:until])
            x_test_len = np.asarray([input_length for _ in range(len(x_test))])

            decode, log = K.ctc_decode(x_test,
                                       x_test_len,
                                       greedy=self.greedy,
                                       beam_width=self.beam_width,
                                       top_paths=self.top_paths)

            probabilities.extend([np.exp(x) for x in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts.extend(np.swapaxes(decode, 0, 1))

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

        return (predicts, probabilities)

    @staticmethod
    def ctc_loss_lambda_func(y_true, y_pred):
        """Function for computing the CTC loss"""

        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)

        # y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
        # output of every model is softmax
        # so sum across alphabet_size_1_hot_encoded give 1
        #               string_length give string length
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

        # y_true strings are padded with 0
        # so sum of non-zero gives number of characters in this string
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        # average loss across all entries in the batch
        loss = tf.reduce_mean(loss)

        return loss


"""
Custom Schedule

Reference:
    Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and
    Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin.
    "Attention Is All You Need", 2017
    arXiv, URL: https://arxiv.org/abs/1706.03762
"""


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom schedule of the learning rate with warmup_steps.
    From original paper "Attention is all you need".
    """

    def __init__(self, d_model, initial_step=0, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, dtype="float32")
        self.initial_step = initial_step
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + self.initial_step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


"""
Networks to the Handwritten Text Recognition Model

Reference:
    Moysset, B. and Messina, R.:
    Are 2D-LSTM really dead for offline text recognition?
    In: International Journal on Document Analysis and Recognition (IJDAR)
    Springer Science and Business Media LLC
    URL: http://dx.doi.org/10.1007/s10032-019-00325-0
"""


def bluche(input_size, d_model):
    """
    Gated Convolucional Recurrent Neural Network by Bluche et al.

    Reference:
        Bluche, T., Messina, R.:
        Gated convolutional recurrent neural networks for multilingual handwriting recognition.
        In: Document Analysis and Recognition (ICDAR), 2017
        14th IAPR International Conference on, vol. 1, pp. 646–651, 2017.
        URL: https://ieeexplore.ieee.org/document/8270042
    """

    input_data = Input(name="input", shape=input_size)
    cnn = Reshape((input_size[0] // 2, input_size[1] // 2, input_size[2] * 4))(input_data)

    cnn = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="tanh")(cnn)

    cnn = Conv2D(filters=16, kernel_size=(2, 4), strides=(2, 4), padding="same", activation="tanh")(cnn)
    cnn = GatedConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn=squeeze_excite_block(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="tanh")(cnn)
    cnn = GatedConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn=squeeze_excite_block(cnn)

    cnn = Conv2D(filters=64, kernel_size=(2, 4), strides=(2, 4), padding="same", activation="tanh")(cnn)
    cnn = GatedConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="tanh")(cnn)
    cnn = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding="valid")(cnn)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=128, activation="tanh")(blstm)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    output_data = Dense(units=d_model, activation="softmax")(blstm)

    return (input_data, output_data)


def puigcerver(input_size, d_model):
    """
    Convolucional Recurrent Neural Network by Puigcerver et al.

    Reference:
        Joan Puigcerver.
        Are multidimensional recurrent layers really necessary for handwritten text recognition?
        In: Document Analysis and Recognition (ICDAR), 2017 14th
        IAPR International Conference on, vol. 1, pp. 67–72. IEEE (2017)

        Carlos Mocholí Calvo and Enrique Vidal Ruiz.
        Development and experimentation of a deep learning system for convolutional and recurrent neural networks
        Escola Tècnica Superior d’Enginyeria Informàtica, Universitat Politècnica de València, 2018
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(input_data)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)
    #cnn=squeeze_excite_block(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)
    #cnn=squeeze_excite_block(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)

    cnn = Dropout(rate=0.2)(cnn)
    cnn = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = LeakyReLU(alpha=0.01)(cnn)
    
    #cnn=cbam_block(input_tensor=cnn,features=80, kernel= 7,spatial= True)

    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(units=d_model, activation="softmax")(blstm)

    return (input_data, output_data)



def flor(input_size, d_model):
    """
    Gated Convolucional Recurrent Neural Network by Flor et al.
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(filters=40, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = Dense(units=256)(bgru)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = Dense(units=d_model, activation="softmax")(bgru)

    return (input_data, output_data)



def a1(input_size, d_model):
    """
    Proposed Model 2 with Gating mechanism s in FLor model
    """

    input_data = Input(name="input", shape=input_size)
    #cnn=PixelAttention2D(input_data.shape[-1])(input_data)
    cnn = Conv2D(filters=8, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="he_uniform",activation='selu')(input_data)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=8, kernel_size=(3, 3), padding="same")(cnn)
    
    

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    
    
    
    cnn = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=24, kernel_size=(3, 3), padding="same")(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    


    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    #cnn=squeeze_excite_block(cnn)

    cnn = Conv2D(filters=40, kernel_size=(3, 3), strides=(2, 4), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(3, 3), strides=(2, 4), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)


    
    
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
    #cnn2= AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
    #cnn=tf.keras.layers.Concatenate()([cnn1, cnn2])
    
    shape = cnn.get_shape()
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)
    #bgru = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)
    #bgru = Dense(units=256)(bgru)

    
    
    #bgru = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)
    bgru = TCN(120,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)

    #bgru = Dense(units=256)(bgru)
    #bgru= TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc2')(bgru)
    

    
    #bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = Dense(units=256)(bgru)

    #bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
  
    output_data = Dense(units=d_model, activation="softmax")(bgru)

    return (input_data, output_data)





def a2(input_size, d_model):
    """
    Proposed Model 2 with Gating mechanism s in Bluche model
    """

    input_data = Input(name="input", shape=input_size)
    #cnn=PixelAttention2D(input_data.shape[-1])(input_data)
    cnn = Conv2D(filters=8, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="he_uniform",activation='selu')(input_data)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = GatedConv2D(filters=8, kernel_size=(3, 3), padding="same")(cnn)
    
    

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = GatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    
    
    
    cnn = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = GatedConv2D(filters=24, kernel_size=(3, 3), padding="same")(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    


    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = GatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    #cnn=squeeze_excite_block(cnn)

    cnn = Conv2D(filters=40, kernel_size=(3, 3), strides=(2, 4), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = GatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = GatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(3, 3), strides=(2, 4), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    #cnn=squeeze_excite_block(cnn)
    cnn = GatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)


    
    
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
    #cnn2= AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
    #cnn=tf.keras.layers.Concatenate()([cnn1, cnn2])
    
    shape = cnn.get_shape()
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)
    #bgru = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)
    #bgru = Dense(units=256)(bgru)

    
    
    #bgru = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)
    bgru = TCN(120,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)

    #bgru = Dense(units=256)(bgru)
    #bgru= TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc2')(bgru)
    

    
    #bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = Dense(units=256)(bgru)

    #bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
  
    output_data = Dense(units=d_model, activation="softmax")(bgru)

    return (input_data, output_data)






def a3(input_size, d_model):
    """
    Flor model with SE Gate Blocks
    """

    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(filters=40, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = Dense(units=256)(bgru)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = Dense(units=d_model, activation="softmax")(bgru)

    return (input_data, output_data)




def shashankbest4senetgateddropouttwo(input_size, d_model):
    """
    Proosed model 2
    """

    input_data = Input(name="input", shape=input_size)
    #cnn=PixelAttention2D(input_data.shape[-1])(input_data)
    cnn = Conv2D(filters=8, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="he_uniform",activation='selu')(input_data)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=8, kernel_size=(3, 3), padding="same")(cnn)
    
    

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    
    
    
    cnn = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=24, kernel_size=(3, 3), padding="same")(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    


    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    #cnn=squeeze_excite_block(cnn)

    cnn = Conv2D(filters=40, kernel_size=(3, 3), strides=(2, 4), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(3, 3), strides=(2, 4), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)


    
    
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
    #cnn2= AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
    #cnn=tf.keras.layers.Concatenate()([cnn1, cnn2])
    
    shape = cnn.get_shape()
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)
    #bgru = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)
    #bgru = Dense(units=256)(bgru)

    
    
    #bgru = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)
    bgru = TCN(120,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)

    #bgru = Dense(units=256)(bgru)
    #bgru= TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc2')(bgru)
    

    
    #bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    #bgru = Dense(units=256)(bgru)

    #bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
  
    output_data = Dense(units=d_model, activation="softmax")(bgru)

    return (input_data, output_data)



def shashankRec(input_size, d_model):
    """
    Proosed model 1
    """

    input_data = Input(name="input", shape=input_size)
    #cnn=PixelAttention2D(input_data.shape[-1])(input_data)
    cnn = Conv2D(filters=8, kernel_size=(3, 3), strides=(1,1), padding="same", kernel_initializer="he_uniform",activation='selu')(input_data)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=8, kernel_size=(3, 3), padding="same")(cnn)
    
    

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    
    
    
    cnn = Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=24, kernel_size=(3, 3), padding="same")(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    


    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)
    #cnn=PixelAttention2D(cnn.shape[-1])(cnn)
    #cnn=squeeze_excite_block(cnn)

    cnn = Conv2D(filters=40, kernel_size=(3, 3), strides=(2, 4), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(3, 3), strides=(2, 4), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn=squeeze_excite_block(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    
    cnn = Dropout(rate=0.2)(cnn)


    
    
    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform",activation='selu')(cnn)
    #cnn = PReLU(shared_axes=[1, 2])(cnn)
    #cnn = selu()(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
    #cnn2= AveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
    #cnn=tf.keras.layers.Concatenate()([cnn1, cnn2])
    
    shape = cnn.get_shape()
    bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)
    #bgru = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)
    #bgru = Dense(units=256)(bgru)

    
    
    #bgru = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)
    bgru = TCN(120,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc1')(bgru)

    #bgru = Dense(units=256)(bgru)
    #bgru= TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = 'wavenet',name = 'tnc2')(bgru)
    

    
    #bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = Dense(units=256)(bgru)

    #bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
  
    output_data = Dense(units=d_model, activation="softmax")(bgru)

    return (input_data, output_data)









def puigcerver_octconv(input_size, d_model):
    """
    Octave CNN by khinggan, architecture is same as puigcerver
    """

    alpha = 0.25
    input_data = Input(name="input", shape=input_size)
    high = input_data
    low = tf.keras.layers.AveragePooling2D(2)(input_data)

    high, low = OctConv2D(filters=16, alpha=alpha)([high, low])
    high = BatchNormalization()(high)
    low = BatchNormalization()(low)
    high = LeakyReLU(alpha=0.01)(high)
    low = LeakyReLU(alpha=0.01)(low)
    high = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(high)
    low = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(low)
    cnn=squeeze_excite_block(cnn)

    high, low = OctConv2D(filters=32, alpha=alpha)([high, low])
    high = BatchNormalization()(high)
    low = BatchNormalization()(low)
    high = LeakyReLU(alpha=0.01)(high)
    low = LeakyReLU(alpha=0.01)(low)
    high = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(high)
    low = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(low)
    cnn=squeeze_excite_block(cnn)

    high = Dropout(rate=0.2)(high)
    low = Dropout(rate=0.2)(low)
    high = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same")(high)
    low = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same")(low)
    high = BatchNormalization()(high)
    low = BatchNormalization()(low)
    high = LeakyReLU(alpha=0.01)(high)
    low = LeakyReLU(alpha=0.01)(low)
    high = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(high)
    low = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(low)

    high = Dropout(rate=0.2)(high)
    low = Dropout(rate=0.2)(low)
    high = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(high)
    low = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(low)
    high = BatchNormalization()(high)
    low = BatchNormalization()(low)
    high = LeakyReLU(alpha=0.01)(high)
    low = LeakyReLU(alpha=0.01)(low)

    high = Dropout(rate=0.2)(high)
    low = Dropout(rate=0.2)(low)
    high = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="same")(high)
    low = Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="same")(low)
    high = BatchNormalization()(high)
    low = BatchNormalization()(low)
    high = LeakyReLU(alpha=0.01)(high)
    low = LeakyReLU(alpha=0.01)(low)

    x = _create_octconv_last_block([high, low], 80, alpha)

    shape = x.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(x)

    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5))(blstm)

    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(units=d_model, activation="softmax")(blstm)

    return (input_data, output_data)


def _create_octconv_last_block(inputs, ch, alpha):
    high, low = inputs

    high, low = OctConv2D(filters=ch, alpha=alpha)([high, low])
    high = BatchNormalization()(high)
    high = Activation("relu")(high)

    low = BatchNormalization()(low)
    low = Activation("relu")(low)

    high_to_high = Conv2D(ch, 3, padding="same")(high)
    low_to_high = Conv2D(ch, 3, padding="same")(low)
    low_to_high = Lambda(lambda x: K.repeat_elements(K.repeat_elements(x, 2, axis=1), 2, axis=2))(low_to_high)

    x = Add()([high_to_high, low_to_high])
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x
    
    
    
def squeeze_excite_block(tensor, ratio=2):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


import random

#print(random.randint(0,9))

class TCN:
    """Creates a TCN layer.
        Args:
            input_layer: A tensor of shape (batch_size, timesteps, input_dim).
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            activation: The activations to use (norm_relu, wavenet, relu...).
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation='norm_relu',
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.2,
                 return_sequences=True,
                 name='tcn'):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation = activation
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding

        # backwards incompatibility warning.
        # o = tcn.TCN(i, return_sequences=False) =>
        # o = tcn.TCN(return_sequences=False)(i)

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' paddings are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(i, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(i)')
            print('Second solution is to pip install keras-tcn==2.1.2 to downgrade.')
            raise Exception()

    def __call__(self, inputs):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16, 32]
        x = inputs
        x = Convolution1D(self.nb_filters, 1, padding=self.padding, name=self.name + '_initial_conv'+str(random.randint(0,99)))(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                x, skip_out = residual_block(x, s, i, self.activation, self.nb_filters,
                                             self.kernel_size, self.padding, self.dropout_rate, name=self.name)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = tf.keras.layers.add(skip_connections)
        x = Activation('relu')(x)

        if not self.return_sequences:
            output_slice_index = -1
            x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)
        return x
        
        
        
def residual_block(x, s, i, activation, nb_filters, kernel_size, padding, dropout_rate=0.2, name=''):
    # type: (Layer, int, int, str, int, int, float, str) -> Tuple[Layer, Layer]
    """Defines the residual block for the WaveNet TCN
    Args:
        x: The previous layer in the model
        s: The stack index i.e. which stack in the overall TCN
        i: The dilation power of 2 we are using for this residual block
        activation: The name of the type of activation to use
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        name: Name of the model. Useful when having multiple TCN.
    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """

    original_x = x
    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding=padding,
                  name=name + str(random.randint(0,99))+'_dilated_conv_%d_tanh_s%d' % (i, s))(x)
    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

    x = SpatialDropout1D(dropout_rate, name=name + str(random.randint(0,99))+'_spatial_dropout1d_%d_s%d_%f' % (i, s, dropout_rate))(x)

    # 1x1 conv.
    x = Convolution1D(nb_filters, 1, padding='same')(x)
    res_x = tf.keras.layers.add([original_x, x])
    return res_x, x
    
    
def wave_net_activation(x):
    # type: (Layer) -> Layer
    """This method defines the activation used for WaveNet
    described in https://deepmind.com/blog/wavenet-generative-model-raw-audio/
    Args:
        x: The layer we want to apply the activation to
    Returns:
        A new layer with the wavenet activation applied
    """
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return multiply([tanh_out, sigm_out])




def expansion_block(x,t,filters,block_id):
    prefix = 'block_{}_'.format(block_id)
    total_filters = t*filters
    x = Conv2D(total_filters,1,padding='same',use_bias=False, name = prefix +'expand')(x)
    x = BatchNormalization(name=prefix +'expand_bn')(x)
    x = ReLU(6,name = prefix +'expand_relu')(x)
    return x

def depthwise_block(x,stride,block_id):
    prefix = 'block_{}_'.format(block_id)
    x = DepthwiseConv2D(3,strides=(stride,stride),padding ='same', use_bias = False, name = prefix + 'depthwise_conv')(x)
    x = BatchNormalization(name=prefix +'dw_bn')(x)
    x = ReLU(6,name=prefix +'dw_relu')(x)
    return x

def projection_block(x,out_channels,block_id):
    prefix = 'block_{}_'.format(block_id)
    x = Conv2D(filters = out_channels,kernel_size = 1,padding='same',use_bias=False,name= prefix + 'compress')(x)
    x = BatchNormalization(name=prefix +'compress_bn')(x)
    return x
def Bottleneck(x,t,filters, out_channels,stride,block_id):
    y = expansion_block(x,t,filters,block_id)
    y = depthwise_block(y,stride,block_id)
    y = projection_block(y, out_channels,block_id)
    if y.shape[-1]==x.shape[-1]:
        y = add([x,y])
    return y
    
    
    
    
    
    
    
    

