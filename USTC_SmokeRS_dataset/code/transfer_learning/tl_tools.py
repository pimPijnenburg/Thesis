import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision

def prepare_transferlearn(transfer_learning_architecture, name = 'Model'): 
    n_classes = 6
    model = models.Sequential(name = name)
    model.add(transfer_learning_architecture)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation= 'relu'))
    model.add(layers.Dense(n_classes, activation= 'softmax', dtype = 'float32'))
    return model


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

def load_data(path='/Users/pimpijnenburg/Desktop/Thesis/USTC_SmokeRS_dataset/data/created_data/for_model_training', batch_size=64):
    train = tf.data.Dataset.load(os.path.join(path, 'train'))
    val = tf.data.Dataset.load(os.path.join(path, 'val'))
    test = tf.data.Dataset.load(os.path.join(path, 'test'))
    
    train = optimize_dataset(train, batch_size, is_training=True)
    val = optimize_dataset(val, batch_size)
    test = optimize_dataset(test, batch_size)
    
    return train, val, test


def create_baseline(architecture, name = 'EfficentNetV2'): 
    n_classes = 6
    output = architecture.output
    output = layers.GlobalAveragePooling2D()(output)
    output = layers.Dense(1024, activation = 'relu')(output)
    output = layers.Dropout(0.5)(output)
    outputs = layers.Dense(n_classes, activation = 'softmax', dtype = 'float32')(output)
    
    model = tf.keras.Model(inputs = architecture.input, outputs = outputs)
    return model