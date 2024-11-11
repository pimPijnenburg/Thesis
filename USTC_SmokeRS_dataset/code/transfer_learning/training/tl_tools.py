import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision, Model, regularizers
from PIL import Image


def setup_mixed_precision():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision policy set to:", policy.name)
    print()



def normalize(image, label): 
    return tf.cast(image, tf.float32) / 255., label



def augment_image(image, label):
    # Random flip 
    image = tf.image.random_flip_left_right(image)
    
    # Random rotation 
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Ensure pixel values are still in [0, 1] range
    image = tf.clip_by_value(image, 0, 1)
    
    return image, label



def process_dataset(dataset, augment = False): 
    dataset = dataset.map(normalize, num_parallel_calls = tf.data.AUTOTUNE)
    
    if augment: 
        dataset = dataset.map(augment_image, num_parallel_calls = tf.data.AUTOTUNE)
    
    #Further data performance optimization
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size = tf.data.AUTOTUNE)
    
    return dataset


def train_val_split(df,val_split = 0.2):

    """
    Splits the trainingset into a training and validation split

    Parameters:
    - df: Training data to split
    - val_splt: Size of the validation 
    """

    #Resetting the generator for reproducible results
    df.reset()

    n_batches = df.num_batches
    batch_size = df.batch_size
    n_samples = df.samples

    print(f'Number of batches in the training data: {n_batches}')
    print(f'Batch size of a single batch {batch_size}')
    print(f'Number of samples in the training dataset {n_samples}')
    print()

    #Setting the size of the train and validation set according to the required split and testing if all batches are included
    val_batches = int(n_batches * val_split)
    train_batches = n_batches - val_batches

    print(f'Number of training data batches with val split of {val_split}: {train_batches}')
    print(f'Number of validation data batches: {val_batches}')
    print()
    assert train_batches + val_batches == n_batches, 'Train and val batches do not add up to total n batches'

    #Iterating through the batches and appending them into lists for train and val
    x_train, y_train = list(), list()
    x_val, y_val = list(), list()

    for batch in range(n_batches):
        x, y = next(df)
        if batch < train_batches:
            x_train.append(x)
            y_train.append(y)

        else:
            x_val.append(x)
            y_val.append(y)

    assert len(x_train) + len(x_val) == n_batches, 'Error in dividing batches into train and val sets'


    #Converting the lists into arrays suited for Tensorflow
    x_train = tf.concat(x_train, axis = 0)
    y_train = tf.concat(y_train, axis = 0)
    x_val = tf.concat(x_val, axis = 0)
    y_val = tf.concat(y_val, axis = 0)

    print(f'Shape of image training set: {x_train.shape}')
    print(f'Shape of image validation set: {x_val.shape}')
    print()
    print(f'Shape of label training set: {y_train.shape}')
    print(f'Shape of label validation set: {y_val.shape}')

    #Testing to see if all the samples are included
    assert x_train.shape[0] + x_val.shape[0] == n_samples, 'Error, not all samples included'



    return x_train, y_train, x_val, y_val



def test_splits(df):

    """
    Converts test data to numpy array
    
    Parameters:
    - df: Testdata
    """
    
    #Resetting the generator for reproducible results
    df.reset()

    n_batches = df.num_batches
    batch_size = df.batch_size
    n_samples = df.samples

    print(f'Number of batches in the test data: {n_batches}')
    print(f'Batch size of a single batch {batch_size}')
    print(f'Number of samples in the test dataset {n_samples}')
    print()

    #Iterating through the batches and appending them into lists for train and val
    x_test, y_test = list(), list()

    for batch in range(n_batches):
        x, y = next(df)
        x_test.append(x)
        y_test.append(y)

    #Converting the lists into arrays suited for Tensorflow
    x_test = tf.concat(x_test, axis = 0)
    y_test = tf.concat(y_test, axis = 0)

    print(f'Shape of image test set: {x_test.shape}')
    print()
    print(f'Shape of label test set: {y_test.shape}')

    #Testing to see if all the samples are included
    assert  x_test.shape[0] == n_samples, 'Error, not all samples included'
    return x_test, y_test



def fc_layers(model, name='smoke_classifier'):
    """Adds same FC layer structure of the baseline to the pretrained architecture. 
    These layers can be trained further upon."""
    x = model.output
    
   #FC1
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    #FC2
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    #output layer
    outputs = layers.Dense(6, activation='softmax', dtype='float32')(x)
    full_model = Model(inputs=model.input, outputs=outputs, name=name)
    
    return full_model





def plot_feature_importances(series, n_plotted): 
    series = series.sort_values(ascending = False).head(n_plotted)
    X = series
    y = series.index
    plt.figure(figsize = (15, 10))
    sns.barplot(x = X, y = y, order = y, orient = 'h', palette = 'viridis', hue = X, legend = False)
    
    plt.title(f'Top {n_plotted} Feature Importance Scores', fontsize = 14)
    plt.xlabel('Importance scores', fontsize = 14)
    plt.ylabel('Feature index', fontsize = 14)
    
    
    plt.tight_layout()
    plt.show()