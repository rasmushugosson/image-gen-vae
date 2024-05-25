
import os
import random
import math

import tensorflow as tf
import tf_keras as tfk
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

tfkl = tfk.layers
tfpl = tfp.layers
tfd = tfp.distributions

import image_gen_vae.constants as consts

def decode_img(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.image.resize(image, [consts.IMAGE_SIZE, consts.IMAGE_SIZE])


@tf.function
def pre_process_image(file_path):
    image = tf.io.read_file(file_path)
    image = decode_img(image)

    return image, image

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=consts.BUFFER_SIZE)
    ds = ds.batch(consts.BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def repeat_images(file_path):
    return tf.data.Dataset.from_tensors(file_path).repeat(consts.DUPLICATE_IMAGES)

def load_datasets(val_percentage):
    print('Loading datasets...')

    folder_path = f'res/images/{consts.IMAGE_SIZE}'

    list_ds = tf.data.Dataset.list_files(str(f'{folder_path}/*'), shuffle=False)
    list_ds = list_ds.shuffle(list_ds.cardinality(), reshuffle_each_iteration=False)

    image_files = os.listdir(folder_path)
    image_count = len(image_files)

    val_size = int(image_count * val_percentage)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    print("Training Images: ", tf.data.experimental.cardinality(train_ds).numpy())
    print("Evaluation Images: ", tf.data.experimental.cardinality(val_ds).numpy())

    if consts.DUPLICATE_IMAGES > 1:
        train_ds = train_ds.repeat(consts.DUPLICATE_IMAGES)
        val_ds = val_ds.repeat(consts.DUPLICATE_IMAGES)
    
    # Immediately calculate the cardinality after duplication and before further transformations
    print("Training Images (post-duplication): ", tf.data.experimental.cardinality(train_ds).numpy())
    print("Validation Images (post-duplication): ", tf.data.experimental.cardinality(val_ds).numpy())

    train_ds = train_ds.map(lambda x: pre_process_image(x), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x: pre_process_image(x), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    return train_ds, val_ds

def load_examples():

    print('Loading examples...')

    folder_path = f'res/images/{consts.IMAGE_SIZE}t'
    files = tf.data.Dataset.list_files(str(f'{folder_path}/*'), shuffle=False).skip(12).take(12)

    images = []

    for file in files:
        image = tf.io.read_file(file)
        image = decode_img(image)
        
        images.append(image)

    return images
