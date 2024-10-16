import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import pandas as pd


def setup_mixed_precision():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision policy set to:", policy.name)

def create_optimizer(learning_rate=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return mixed_precision.LossScaleOptimizer(optimizer) 
    

AUTOTUNE = tf.data.AUTOTUNE

def optimize_dataset(dataset, batch_size, is_training=False):
    dataset = dataset.batch(batch_size)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset




def load_and_prepare_data(path, batch_size=32):
    # Load the dataframes
    train_df = pd.read_parquet(os.path.join(path, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(path, 'val.parquet'))
    test_df = pd.read_parquet(os.path.join(path, 'test.parquet'))

    def preprocess_image(red, green, blue, label):
        image = tf.stack([red, green, blue], axis=-1)
        image = tf.reshape(image, (256, 256, 3))
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, depth=6)
        return image, label

    def create_dataset(df, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((
            df['red_channel'].tolist(),
            df['green_channel'].tolist(),
            df['blue_channel'].tolist(),
            df['class'].tolist()
        ))
        
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df, shuffle=False)
    test_dataset = create_dataset(test_df, shuffle=False)

    return train_dataset, val_dataset, test_dataset





