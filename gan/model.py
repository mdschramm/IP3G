#!/usr/bin/env python
# coding: utf-8

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from preprocessing.filter_utils import filter_classes
# import imageio
# import datetime

LATENT_DIM = 128
C_CAT_DIM = 54
NUM_CHANNELS = 1
BATCH_SIZE = 64
IMAGE_SIZE = 128
EPOCHS = 2000
RUN_MODE = os.environ.get("RUN_MODE", "local")

EXCLUDED_CLASSES = [6, 24, 25, 31]

PREPROCESSING_DIR = "output/preprocessing"
DATA_DIR = f"output/gan/{RUN_MODE}"
GENERATOR_MODEL_FILE = f"{DATA_DIR}/generator.keras"
DISCRIMINATOR_MODEL_FILE = f"{DATA_DIR}/discriminator.keras"
Q_NETWORK_MODEL_FILE = f"{DATA_DIR}/q_network.keras"


def initialize_dataset():

    DATA_FILE = f"{PREPROCESSING_DIR}/resized_expressions.npy"
    LABEL_FILE = f"{PREPROCESSING_DIR}/y_primary_disease_or_tissue.npy"
    x_train = np.load(DATA_FILE)
    y_train = np.load(LABEL_FILE).astype("float32")
    x_train, _ = filter_classes(x_train, y_train, EXCLUDED_CLASSES)

    # -1 to 1 normalization
    x_train = x_train * 2.0 - 1.0
    x_train = x_train.astype("float32")
    x_train = np.reshape(x_train, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((x_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    print(f"Shape of training images: {x_train.shape}")
    return dataset


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
  q_net_out = layers.Dense(C_CAT_DIM , activation='softmax')(q_net_out)
  q_model = keras.models.Model(img_input, q_net_out, name='q_network')

  return d_model, q_model

def get_generator_model():
  noise = layers.Input(shape=(LATENT_DIM,))
  labels = layers.Input(shape=(C_CAT_DIM,))
  inputs =layers.concatenate([noise,labels], axis=1)
  x = layers.Dense(8 * 8 * (LATENT_DIM+C_CAT_DIM), name='gen_l1')(inputs)
  x = layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Reshape((8, 8, LATENT_DIM+C_CAT_DIM))(x)
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
        self.d_steps = 5
        self.gp_weight = 100

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer,q_optimizer, d_loss_fn, g_loss_fn, q_loss_fn):
        super(WINFOGAN, self).compile(jit_compile=False)
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

        # verify this is same as global BATCH_SIZE
        batch_size = tf.shape(real_images)[0]

        for i in range(self.d_steps):
          random_latent_vectors = tf.random.normal(
              shape=(batch_size, self.latent_dim)
          )
          indx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=C_CAT_DIM, dtype=tf.int32)
          labels = tf.one_hot(indx , C_CAT_DIM)

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
        indx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=C_CAT_DIM, dtype=tf.int32)
        labels = tf.one_hot(indx , C_CAT_DIM)
        
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
    def __init__(self, latent_dim=LATENT_DIM):
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        # Sample noise for the interpolation.
        _noise = tf.random.normal(shape=(4, LATENT_DIM))
        _label = keras.utils.to_categorical([0,1,2,3], C_CAT_DIM)
        _label = tf.cast(_label, tf.float32)

        # Combine the noise and the labels and run inference with the generator.
        fake_images = self.model.generator.predict([_noise, _label])
        fake_images = fake_images * 0.5 + 0.5
        fake_images *= 255.0
        converted_images = fake_images.astype(np.uint8)
        converted_images = tf.image.resize(converted_images, (256, 256)).numpy().astype(np.uint8)


        for i in range(4):
          plt.subplot(2,2,i+1)
          plt.imshow(converted_images[i][:,:,0],cmap='gray')
        
        # Save to gan_monitor_images subfolder
        output_dir = os.path.join(DATA_DIR, "gan_monitor_images")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"epoch_{epoch:04d}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved GAN monitor image to {output_path}")


def plot_training_history(history, output_path=f"{DATA_DIR}/gan_training_history.png"):
    """Plot GAN training metrics over time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Generator loss (use 'g_loss' key from train_step return)
    axes[0].plot(history.history['g_loss'])
    axes[0].set_title('Generator Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Discriminator loss (use 'd_loss' key from train_step return)
    axes[1].plot(history.history['d_loss'])
    axes[1].set_title('Discriminator Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    # Q-network loss (use 'q_loss' key from train_step return)
    axes[2].plot(history.history['q_loss'])
    axes[2].set_title('Q-Network Loss (Class Discovery)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Training history plot saved to {output_path}")


def analyze_training(history):
    """Analyze training for common GAN problems."""
    # Use correct keys from train_step return dict
    gen_loss = history.history['g_loss']
    disc_loss = history.history['d_loss']
    q_loss = history.history['q_loss']
    
    print("\n" + "="*60)
    print("TRAINING ANALYSIS")
    print("="*60)
    
    # Mode collapse: generator loss stuck at high value
    if gen_loss[-1] > gen_loss[len(gen_loss)//2]:
        print("⚠️  Warning: Generator loss increasing - possible mode collapse")
    
    # Discriminator too strong: disc loss near 0
    if disc_loss[-1] < 0.1:
        print("⚠️  Warning: Discriminator too strong - generator may struggle")
    
    # Q-network not learning: q_loss not decreasing
    if q_loss[-1] > q_loss[len(q_loss)//4]:
        print("⚠️  Warning: Q-network not learning classes well")
    
    # Good training
    if gen_loss[-1] < gen_loss[0] and 0.1 < disc_loss[-1] < 2.0:
        print("✅ Training looks healthy!")
    
    print(f"\nFinal metrics:")
    print(f"  Generator loss: {gen_loss[-1]:.4f}")
    print(f"  Discriminator loss: {disc_loss[-1]:.4f}")
    print(f"  Q-network loss: {q_loss[-1]:.4f}")
    
    print(f"\nInitial vs Final:")
    print(f"  Generator: {gen_loss[0]:.4f} → {gen_loss[-1]:.4f} ({((gen_loss[-1]-gen_loss[0])/gen_loss[0]*100):+.1f}%)")
    print(f"  Discriminator: {disc_loss[0]:.4f} → {disc_loss[-1]:.4f} ({((disc_loss[-1]-disc_loss[0])/disc_loss[0]*100):+.1f}%)")
    print(f"  Q-network: {q_loss[0]:.4f} → {q_loss[-1]:.4f} ({((q_loss[-1]-q_loss[0])/q_loss[0]*100):+.1f}%)")
    print("="*60 + "\n")


def train_gan(info_gan, dataset, epochs=EPOCHS):
    callback = GANMonitor(LATENT_DIM)
    history = info_gan.fit(dataset, epochs=epochs, callbacks=[callback])
    
    # CRITICAL: Save models IMMEDIATELY after training, before any plotting
    print("\n" + "="*60)
    print("SAVING MODELS (before plotting to prevent data loss)")
    print("="*60)
    save_gan_components(info_gan)
    print("✅ Models saved successfully!\n")
    
    # Save training history
    history_path = f"{DATA_DIR}/gan_training_history.npy"
    np.save(history_path, history.history)
    print(f"Training history saved to {history_path}")
    
    # Plot training curves (safe to fail now that models are saved)
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"⚠️  Warning: Failed to plot training history: {e}")
    
    # Analyze training (safe to fail)
    try:
        analyze_training(history)
    except Exception as e:
        print(f"⚠️  Warning: Failed to analyze training: {e}")
    
    return info_gan

def save_gan_components(info_gan):
    print("Saving GAN components")
    # Save only the component models, not the custom WINFOGAN wrapper
    info_gan.generator.save(GENERATOR_MODEL_FILE)
    info_gan.discriminator.save(DISCRIMINATOR_MODEL_FILE)
    info_gan.q_network.save(Q_NETWORK_MODEL_FILE)

def compile_info_gan(g_model, d_model, q_network):
    print("Compiling InfoGAN")
    info_gan = WINFOGAN(
        discriminator=d_model, generator=g_model, q_network=q_network, latent_dim=LATENT_DIM
    )
    info_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003,beta_1=0.5, beta_2=0.9),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003,beta_1=0.5, beta_2=0.9),
        q_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9),
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
        q_loss_fn=keras.losses.CategoricalCrossentropy()
    )
    return info_gan


def load_or_get_component_models(refresh=False):
    if refresh:
        print("Refresh: True, re-training models")
        d_model, q_network = get_discriminator_model()
        g_model = get_generator_model()
        return g_model, d_model, q_network
        
    try:
        print("Refresh: False, loading models")
        g_model = keras.models.load_model(GENERATOR_MODEL_FILE)
        d_model = keras.models.load_model(DISCRIMINATOR_MODEL_FILE)
        q_network = keras.models.load_model(Q_NETWORK_MODEL_FILE)
        return g_model, d_model, q_network
    except:
        print("Refresh: False, models not found, re-training")
        d_model, q_network = get_discriminator_model()
        g_model = get_generator_model()
        return g_model, d_model, q_network

def load_or_train_gan(dataset, refresh=False, epochs=EPOCHS):
    g_model, d_model, q_network = load_or_get_component_models(refresh)
    info_gan = compile_info_gan(g_model, d_model, q_network)
    if refresh:
        train_gan(info_gan, dataset, epochs=epochs)  # train_gan now saves models internally
    return info_gan

def generate_synthetic_dataset(generator, num_samples_per_class=100, output_dir=DATA_DIR):
    """
    Generate synthetic images for all latent classes using the trained GAN.
    
    NOTE: This GAN is UNSUPERVISED - it discovers latent classes without labels.
    The class IDs here (0-53) do NOT correspond to real phenotype labels.
    You must map them to real phenotypes by evaluating with the classifier.
    
    Args:
        generator: Trained generator model
        num_samples_per_class: Number of images to generate per latent class
        output_dir: Directory to save generated data
        
    Returns:
        X_synthetic: Generated images, shape (num_classes * num_samples_per_class, 128, 128)
        y_synthetic: One-hot encoded latent class labels, shape (num_classes * num_samples_per_class, num_classes)
    """
    print(f"\nGenerating synthetic dataset: {num_samples_per_class} samples per latent class")
    print(f"Total samples: {C_CAT_DIM * num_samples_per_class}")
    print(f"\nIMPORTANT: These are UNSUPERVISED latent classes, not real phenotype labels!")
    print(f"Use evaluation.py to map latent classes to real phenotypes.\n")
    
    all_images = []
    all_labels = []
    
    for class_id in range(C_CAT_DIM):
        # Generate random noise
        noise = np.random.normal(0, 1, (num_samples_per_class, LATENT_DIM))
        
        # Create one-hot encoded labels for this latent class
        labels = np.zeros((num_samples_per_class, C_CAT_DIM))
        labels[:, class_id] = 1
        
        # Generate images
        generated_images = generator.predict([noise, labels], verbose=0)
        
        # Convert from [-1, 1] back to [0, 1] range (reverse the normalization)
        generated_images = (generated_images + 1.0) / 2.0
        
        # Remove channel dimension for consistency with real data
        generated_images = generated_images.squeeze(axis=-1)
        
        all_images.append(generated_images)
        all_labels.append(labels)
        
        if (class_id + 1) % 10 == 0:
            print(f"  Generated {class_id + 1}/{C_CAT_DIM} latent classes")
    
    # Concatenate all classes
    X_synthetic = np.concatenate(all_images, axis=0)
    y_synthetic = np.concatenate(all_labels, axis=0)
    
    # Shuffle the dataset
    indices = np.arange(len(X_synthetic))
    np.random.shuffle(indices)
    X_synthetic = X_synthetic[indices]
    y_synthetic = y_synthetic[indices]
    
    # Save to disk
    synthetic_features_path = f"{output_dir}/synthetic_resized_expressions.npy"
    synthetic_labels_path = f"{output_dir}/synthetic_latent_classes.npy"
    
    np.save(synthetic_features_path, X_synthetic)
    np.save(synthetic_labels_path, y_synthetic)
    
    print(f"\nSynthetic dataset saved:")
    print(f"  Features: {synthetic_features_path}")
    print(f"  Latent class labels: {synthetic_labels_path}")
    print(f"  Shape: {X_synthetic.shape}")
    print(f"  Value range: [{X_synthetic.min():.4f}, {X_synthetic.max():.4f}]")
    print(f"\nNext steps:")
    print(f"  1. Run classifier on synthetic data to see what phenotypes it predicts")
    print(f"  2. Map latent classes to real phenotypes based on classifier predictions")
    print(f"  3. Compare GAN-discovered classes vs. ground truth phenotypes")
    
    return X_synthetic, y_synthetic


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train unsupervised InfoGAN and generate synthetic gene expression images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Train GAN from scratch (default 2000 epochs)
        python model.py --train --refresh
        
        # Train with custom number of epochs
        python model.py --train --refresh --epochs 1000
        
        # Load existing GAN and generate synthetic data
        python model.py --generate --samples-per-class 100
        
        # Train and generate in one command
        python model.py --train --refresh --generate --samples-per-class 200 --epochs 2000
        """
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the GAN (loads existing model unless --refresh is set)'
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Retrain the GAN from scratch (only used with --train)'
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate synthetic dataset after training'
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=100,
        help='Number of synthetic samples to generate per latent class (default: 100)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.train and not args.generate:
        parser.error("Must specify at least one of: --train, --generate")
    
    if args.refresh and not args.train:
        parser.error("--refresh can only be used with --train")
    
    os.makedirs(DATA_DIR, exist_ok=True)

    info_gan = None

    # Train GAN if requested
    if args.train:
        print("=" * 60)
        print("TRAINING UNSUPERVISED INFOGAN")
        print("=" * 60)
        print(f"Training for {args.epochs} epochs")
        dataset = initialize_dataset()
        info_gan = load_or_train_gan(dataset, refresh=args.refresh, epochs=args.epochs)
        info_gan.summary()
        print("\nTraining complete!")
    
    # Generate synthetic dataset if requested
    if args.generate:
        print("\n" + "=" * 60)
        print("GENERATING SYNTHETIC DATASET")
        print("=" * 60)
        
        # Load GAN if not already loaded from training
        if info_gan is None:
            print("Loading existing GAN models...")
            dataset = initialize_dataset()
            info_gan = load_or_train_gan(dataset, refresh=False)
        
        generator = info_gan.generator
        generate_synthetic_dataset(
            generator,
            num_samples_per_class=args.samples_per_class,
            output_dir=DATA_DIR
        )
        print("\nGeneration complete!")