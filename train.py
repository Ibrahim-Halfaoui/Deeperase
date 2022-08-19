import click
from pathlib import Path
from typing import Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import deeperase


def _show_batch(input_images: tf.Tensor, target_images: tf.Tensor):
    cols = 2
    rows = input_images.shape[0]
    i = 1

    for inp, target in zip(input_images, target_images):
        plt.subplot(rows, cols, i)
        plt.title('Input Image (Noise)')
        plt.imshow(inp)
        plt.axis('off')

        plt.subplot(rows, cols, i + 1)
        plt.title('Target (Clean)')
        plt.imshow(target)
        plt.axis('off')

        i += 2

    plt.show()
    plt.close()


def _build_model(input_shape: Tuple[int, int, int], 
                 is_gan: bool) -> tf.keras.Model:

    unet = deeperase.build_unet(input_shape, 
                                backbone='resnet50',
                                n_channels=1,
                                activation='relu',
                                upsample_strategy='conv',
                                name='unet_generator')
    unet_optimizer = tf.optimizers.Adam(learning_rate=9e-4, beta_1=.5)

    if is_gan:
        discriminator = deeperase.build_discriminator(input_shape, 
                                                      name='discriminator')
        discriminator_optimizer = tf.optimizers.Adam(learning_rate=5e-4,
                                                     beta_1=.5)
    else:
        discriminator = None
        discriminator_optimizer = None

    model = deeperase.DeepErase(generator=unet, discriminator=discriminator)
    model.build((None,) + input_shape)
    model.compile(optimizer=unet_optimizer, 
                  discriminator_optimizer=discriminator_optimizer)
    return model


@click.command()
@click.option('-b', '--batch_size', default=4,
              help='Batch_size used')
@click.option('--gan/--no-gan', default=False, 
              help='If this flag is set to True, then the DeepEraser model '
                   'will use a discriminator alongside an adversarial loss ' 
                   'to improve the generated images. Otherwise, the model will '
                   'only consist of U-Net generator')
@click.option('--logdir', type=click.Path(file_okay=False), default='logs',
              help='Tensorboard logdir. The script creates a subfolder with '
                   'the current datetime to differentiate experiments.')
@click.option('--epochs', default=50, type=int, 
              help='Times to cycle through all the dataset at training time.')
@click.option('--steps_per_epoch', default=50, type=int,
              help='Times to cycle through one epoch at training time.')
@click.option('--resume', type=click.Path(file_okay=False), default=None,
              help='Start training from an existing model checkpoint')
@click.option('--checkpoint-dir', type=click.Path(file_okay=False), 
              default='models/deeperase',
              help='Directory to save model checkpoint after every epoch')
def train(
          logdir: str,
          gan: bool, 
          epochs: int, batch_size: int, steps_per_epoch: int,
          resume: str, checkpoint_dir: str):

    # Create logdir subdirectory
    logdir_suffix = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logdir = Path(logdir) / logdir_suffix

    image_size = (64, 256)
    total_iter = batch_size * steps_per_epoch * epochs
    train_ds, test_ds = deeperase.data.build_dataset_relie(
                                                     image_size=image_size,
                                                     total_iter=total_iter,
                                                     batch_size=batch_size
                                                    )

    # _show_batch(*next(iter(train_ds.batch(4))))
    model = _build_model(image_size + (1,), gan)
    model.summary()

    if resume is not None:
        model.load_weights(resume)

    model.fit(train_ds.prefetch(tf.data.AUTOTUNE).batch(batch_size),
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=test_ds.batch(batch_size),
              callbacks=[
                  deeperase.callbacks.TensorBoard(logdir=logdir,
                                                  images_dataset=test_ds),
                  tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir)])


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train()
