import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, models

def prepare_transferlearn(transfer_learning_architecture, name = 'Model'): 
    n_classes = 6
    model = models.Sequential(name = name)
    model.add(transfer_learning_architecture)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation= 'relu'))
    model.add(layers.Dense(n_classes, activation= 'softmax'))
    return model



def load_data(path = '/Users/pimpijnenburg/Desktop/Thesis/USTC_SmokeRS_dataset/data/created_data/for_model_training', batch_size = 64): 
    train = tf.data.Dataset.load(os.path.join(path, 'train')).batch(batch_size= batch_size)
    val = tf.data.Dataset.load(os.path.join(path, 'val')).batch(batch_size= batch_size)
    test = tf.data.Dataset.load(os.path.join(path, 'test')).batch(batch_size= batch_size)
    
    return train, val, test