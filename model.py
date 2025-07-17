#!/usr/bin/env python
# coding: utf-8

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# import imageio
# import datetime




batch_size = 64
num_channels = 1
image_size = 128
latent_dim = 128


# DATA_FILE = f"{root}data.npy"
DATA_FILE = "sample_data.npy"
x_train = np.load(DATA_FILE)

# -1 to 1 normalization
x_train = x_train * 2.0 - 1.0
c_cat_dim = 54
x_train = x_train.astype("float32")
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])




# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((x_train))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {x_train.shape}")


def get_discriminator_model():
  img_input = layers.Input(shape=(128, 128, 1))
  x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same",name='disc_l1')(img_input)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.GlobalMaxPooling2D()(x)
  disc_out = layers.Dense(1, name='disc_out')(x)

  d_model = keras.models.Model(img_input, disc_out, name="discriminator")

  q_net_out = layers.Dense(128, activation='relu', kernel_initializer='he_normal' , bias_initializer='he_normal'  )(x)
  q_net_out = layers.Dense(c_cat_dim , activation='softmax')(q_net_out)
  q_model = keras.models.Model(img_input, q_net_out, name='q_network')

  return d_model, q_model




def get_generator_model():
  noise = layers.Input(shape=(latent_dim,))
  labels = layers.Input(shape=(c_cat_dim,))
  inputs =layers.concatenate([noise,labels], axis=1)
  x = layers.Dense(8 * 8 * (latent_dim+c_cat_dim), name='gen_l1')(inputs)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Reshape((8, 8, latent_dim+c_cat_dim))(x)
  x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same",use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same",use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same",use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2DTranspose(128, (7, 7), strides=(2, 2), padding="same",use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2D(1, (7, 7), padding="same",use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('tanh',name='gen_out')(x)
  g_model = keras.models.Model([noise,labels], x, name="generator")
  return g_model




d_model, q_network = get_discriminator_model()
d_model.summary()
# d_model = tf.keras.models.load_model('/content/drive/MyDrive/Ravaee/GTEX/GE W-InfoGAN Discriminator.h5')
# q_network = tf.keras.models.load_model('/content/drive/MyDrive/Ravaee/GTEX/GE W-InfoGAN q-net.h5')




g_model = get_generator_model()
g_model.summary()
# g_model = tf.keras.models.load_model('/content/drive/MyDrive/Ravaee/GTEX/GE W-InfoGAN Generator.h5')




class WINFOGAN(keras.Model):
    def __init__(self, discriminator, generator,q_network, latent_dim):
        super(WINFOGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.q_network = q_network
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.q_loss_tracker = keras.metrics.Mean(name="q_loss")
        self.batch_size = batch_size
        self.d_steps = 5
        self.gp_weight = 100

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer,q_optimizer, d_loss_fn, g_loss_fn, q_loss_fn):
        super(WINFOGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.q_optimizer = q_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.q_loss_fn = q_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        # Unpack the data.
        real_images = data

        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
          random_latent_vectors = tf.random.normal(
              shape=(batch_size, self.latent_dim)
          )
          indx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=c_cat_dim, dtype=tf.int32)
          labels = tf.one_hot(indx , c_cat_dim)

          # Train the discriminator.
          with tf.GradientTape() as tape:
              self.discriminator.trainable = True
              # Generate fake images from the latent vector
              fake_images = self.generator([random_latent_vectors,labels], training=True)
              # Get the logits for the fake images
              fake_logits = self.discriminator(fake_images, training=True)
              # Get the logits for the real images
              real_logits = self.discriminator(real_images, training=True)

              # Calculate the discriminator loss using the fake and real image logits
              d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
              # Calculate the gradient penalty
              gp = self.gradient_penalty(batch_size, real_images, fake_images)
              # Add the gradient penalty to the original discriminator loss
              d_loss = d_cost + gp * self.gp_weight

          # Get the gradients w.r.t the discriminator loss
          d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
          # Update the weights of the discriminator using the discriminator optimizer
          self.d_optimizer.apply_gradients(
              zip(d_gradient, self.discriminator.trainable_variables)
          )

         # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        indx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=c_cat_dim, dtype=tf.int32)
        labels = tf.one_hot(indx , c_cat_dim)
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as qn_tape:
            self.discriminator.trainable = False
            
            # NOt needed as gradienttape automatically records the trainable variables
            # g_tape.watch(self.generator.trainable_variables)
            # qn_tape.watch(self.q_network.trainable_variables)
            
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors,labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            
            cat_output = self.q_network(generated_images, training=True)
            cat_loss = self.q_loss_fn(labels , cat_output)
            
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits) + cat_loss

        # Get the gradients w.r.t the generator loss
        gen_gradient = g_tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        
        qn_gradinet = qn_tape.gradient(cat_loss , self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(
            zip(qn_gradinet , self.q_network.trainable_variables))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        self.q_loss_tracker.update_state(cat_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "q_loss": self.q_loss_tracker.result()
        }




def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)




class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        # Sample noise for the interpolation.
        _noise = tf.random.normal(shape=(4, latent_dim))
        _label = keras.utils.to_categorical([0,1,2,3], c_cat_dim)
        _label = tf.cast(_label, tf.float32)

        # Combine the noise and the labels and run inference with the generator.
        fake_images = self.model.generator.predict([_noise, _label])
        fake_images = fake_images * 0.5 + 0.5
        fake_images *= 255.0
        converted_images = fake_images.astype(np.uint8)
        # converted_images = tf.image.resize(converted_images, (256, 256)).numpy().astype(np.uint8)

        
        for i in range(4):
          plt.subplot(2,2,i+1)
          plt.imshow(converted_images[i][:,:,0],cmap='gray')
        plt.show()




callback = GANMonitor(latent_dim=latent_dim)




info_gan = WINFOGAN(
    discriminator=d_model, generator=g_model, q_network=q_network, latent_dim=latent_dim
)
info_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003,beta_1=0.5, beta_2=0.9),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003,beta_1=0.5, beta_2=0.9),
    q_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9),
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
    q_loss_fn=keras.losses.CategoricalCrossentropy()
)



print("stuff")
# history = info_gan.fit(dataset, epochs=2000 , callbacks=[callback])
history = info_gan.fit(dataset, epochs=100 , callbacks=[callback])

# cond_gan.fit(dataset, epochs=20)

# Save info_gan model weights
info_gan.generator.save('generator.h5')
info_gan.discriminator.save('discriminator.h5')
info_gan.q_network.save('q_network.h5')
info_gan.save('info_gan.h5')






