from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.models import Model
from keras.losses import mean_squared_error
from keras.layers import BatchNormalization
from keras.layers import Input, Lambda, Flatten, Concatenate, Reshape, Activation
from keras.layers import Dense, Conv2D, Conv2DTranspose
import numpy as np


# Sampling Layer -------------------------------------------------------------------------------------------------------


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# Auto Encoder ---------------------------------------------------------------------------------------------------------


class AutoEncoder:

    def __init__(self, input_shape, z_dim=10, c_dim=1, beta=1., **kwargs):
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.beta = beta
        self.kwargs = kwargs

        self._model()
        self._loss()
        self._compile()

    def _model(self):
        # Encoder
        self.inputs = Input(shape=self.input_shape)
        h = Conv2D(64, 5, strides=2, padding='same')(self.inputs)
        h = Activation('relu')(h)
        h = Conv2D(128, 5, strides=2, padding='same')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Activation('relu')(h)
        h = Conv2D(256, 5, strides=2, padding='same')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Activation('relu')(h)
        h = Flatten()(h)
        self.z_mean = Dense(self.z_dim)(h)
        self.z_log_var = Dense(self.z_dim)(h)
        self.c = Dense(self.c_dim)(h)

        # Sample z
        self.z = Lambda(sampling, output_shape=(self.z_dim,))([self.z_mean, self.z_log_var])

        # Encoder model
        self.encoder = Model(self.inputs, [self.z_mean, self.z_log_var, self.z, self.c])

        # Decoder
        latent_inputs = Input(shape=(self.z_dim,))
        info_inputs = Input(shape=(self.c_dim,))
        h = Concatenate()([latent_inputs, info_inputs])
        h = Dense(self.input_shape[0] * self.input_shape[1] // 2 ** 6 * 256, activation='relu')(h)
        h = Reshape((self.input_shape[0] // 2 ** 3, self.input_shape[1] // 2 ** 3, 256))(h)
        h = Conv2DTranspose(256, 5, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2DTranspose(128, 5, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2DTranspose(64, 5, strides=2, padding='same')(h)
        h = Activation('relu')(h)
        decoder_outputs = Conv2D(self.input_shape[2], 5, padding='same')(h)  # linear activation

        # Decoder model
        self.decoder = Model([latent_inputs, info_inputs], decoder_outputs)

        # Auto encoder model
        self.outputs = self.decoder(self.encoder(self.inputs)[2:])
        self.cvae = Model(self.inputs, self.outputs)

    def _loss(self):

        # Reconstruction
        reconstruction_loss = K.mean(mean_squared_error(self.inputs, self.outputs))
        self.reconstruction_loss = reconstruction_loss * np.prod(self.input_shape)

        # KL
        kl_loss = 1. + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        self.kl_loss = self.beta * K.mean(kl_loss)

        # Total
        _vae_loss = self.reconstruction_loss + self.kl_loss
        self.vae_loss = lambda y_true, y_pred: _vae_loss  # wrapper for keras gen to work

    def _compile(self):
        lr = self.kwargs['learning_rate'] if 'learning_rate' in self.kwargs else 0.002
        self.cvae.compile(optimizer=RMSprop(lr=lr), loss=self.vae_loss)
        self.cvae.metrics_tensors = [self.reconstruction_loss, self.kl_loss]
        self.cvae.metrics_names = ['reconstruction', 'weighted kl']

    def restore(self):
        self.cvae.load_weights('cvae.h5')

    def train(self, train_dir, val_dir, epochs=50, batch_size=128, save_dir='./results'):
        # Generators, add the latest to the class
        color_mode = 'rgb' if self.input_shape[-1] > 1 else 'grayscale'
        datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='constant')
        self.train_gen = datagen.flow_from_directory(train_dir,
                                                     target_size=self.input_shape[:2],
                                                     color_mode=color_mode,
                                                     class_mode='categorical',
                                                     batch_size=batch_size)
        self.val_gen = datagen.flow_from_directory(val_dir,
                                                   target_size=self.input_shape[:2],
                                                   color_mode=color_mode,
                                                   class_mode='categorical',
                                                   batch_size=batch_size)

        # Fit
        steps_per_epoch = np.ceil(self.train_gen.n / batch_size)
        validation_steps = np.ceil(self.val_gen.n / batch_size)
        self.cvae.fit_generator(self.train_gen,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=self.val_gen,
                                validation_steps=validation_steps)

        # Save weights
        self.cvae.save_weights(save_dir + '/cvae.h5')

    def predict(self, type='cvae', **kwargs):
        # Check availability
        assert type in ['cvae', 'decoder', 'encoder'], \
            'Only able to predict cvae, decoder, or encoder. You chose: %r' % type

        if type == 'cvae':
            return self.cvae.predict(kwargs['x'])
        elif type == 'decoder':
            return self.decoder.predict([kwargs['z'], kwargs['c']])
        else:
            return self.encoder.predict(kwargs['x'])

    def sample_data(self, type='val', size=100):
        # Check availability
        assert type in ['val', 'train'], \
            'Only able to sample from val or train. You chose: %r' % type

        if type == 'val':
            assert hasattr(self, 'val_gen'), 'No val data attached to model.'
            gen = self.val_gen
        else:
            assert hasattr(self, 'train_gen'), 'No train data attached to model.'
            gen = self.train_gen

        # Single element per batch
        gen.batch_size = 1

        # Data
        data = []
        for (x, _) in gen:
            data.append(x[0])
            if len(data) > size - 1:
                break

        return np.asarray(data)
