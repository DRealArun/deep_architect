
import numpy as np
import scipy as sp
import tensorflow as tf
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import gc
import os, sys, tarfile, urllib
import scipy.io as sio
from scipy.misc import *
import argparse
import glob
from PIL import Image
import random

class InMemoryDataset:
    """Wrapper around a dataset for iteration that allows cycling over the
    dataset.

    This functionality is especially useful for training. One can specify if
    the data is to be shuffled at the end of each epoch. It is also possible
    to specify a transformation function to applied to the batch before
    being returned by next_batch.

    """

    def __init__(self, X, y, shuffle_at_epoch_begin, batch_transform_fn=None):
        if X.shape[0] != y.shape[0]:
            assert ValueError("X and y the same number of examples.")

        self.X = X
        self.y = y
        self.shuffle_at_epoch_begin = shuffle_at_epoch_begin
        self.batch_transform_fn = batch_transform_fn
        self.iter_i = 0

    def get_num_examples(self):
        return self.X.shape[0]

    def next_batch(self, batch_size):
        """Returns the next batch in the dataset.

        If there are fewer that batch_size examples until the end
        of the epoch, next_batch returns only as many examples as there are
        remaining in the epoch.

        """
        #print("Batch Size :",batch_size)
        n = self.X.shape[0]
        i = self.iter_i
        # shuffling step.
        if i == 0 and self.shuffle_at_epoch_begin:
            inds = np.random.permutation(n)
            print("Value of inds = ",inds,len(inds))
            print("Batch Size :",batch_size)
            gc.collect()
            for i in range(20):
            	pass
            gc.collect()
            self.X = self.X[inds]
            self.y = self.y[inds]

        # getting the batch.
        eff_batch_size = min(batch_size, n - i)
        X_batch = self.X[i:i + eff_batch_size]
        y_batch = self.y[i:i + eff_batch_size]
        self.iter_i = (self.iter_i + eff_batch_size) % n

        # transform if a transform function was defined.
        if self.batch_transform_fn != None:
            X_batch_out, y_batch_out = self.batch_transform_fn(X_batch, y_batch)
        else:
            X_batch_out, y_batch_out = X_batch, y_batch

        return (X_batch_out, y_batch_out)

def load_mnist(data_dir, flatten=False, one_hot=True, normalize_range=False,
        whiten_pixels=False, border_pad_size=0):
    from tensorflow.examples.tutorials.mnist import input_data
    # print data_dir
    mnist = input_data.read_data_sets(data_dir, one_hot=one_hot, reshape=flatten, validation_size=6000)

    def _extract_fn(x):
        X = x.images
        y = x.labels
        y = y.astype('float32')

        if not normalize_range:
            X *= 255.0

        return (X, y)

    Xtrain, ytrain = _extract_fn(mnist.train)
    Xval, yval = _extract_fn(mnist.validation)
    Xtest, ytest = _extract_fn(mnist.test)
    print(Xtrain.shape)
    if whiten_pixels:
        mean = Xtrain.mean()
        std = Xtrain.std()
        print(mean,std)
        Xtrain = (Xtrain - mean) / std
        Xval = (Xval - mean) / std
        Xtest = (Xtest - mean) / std

    # NOTE: the zero padding is done after the potential whitening
    if border_pad_size > 0:
        Xtrain = zero_pad_border(Xtrain, border_pad_size)
        Xval = zero_pad_border(Xval, border_pad_size)
        Xtest = zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def load_fashion(data_dir, flatten=False, one_hot=True, normalize_range=False,
        whiten_pixels=False, border_pad_size=0):
    from tensorflow.examples.tutorials.mnist import input_data
    # print data_dir
    mnist = input_data.read_data_sets(data_dir, one_hot=one_hot, reshape=flatten, validation_size=6000)

    def _extract_fn(x):
        X = x.images
        y = x.labels
        y = y.astype('float32')

        if not normalize_range:
            X *= 255.0

        return (X, y)

    Xtrain, ytrain = _extract_fn(mnist.train)
    Xval, yval = _extract_fn(mnist.validation)
    Xtest, ytest = _extract_fn(mnist.test)
    print(Xtrain.shape)
    if whiten_pixels:
        mean = Xtrain.mean()
        std = Xtrain.std()
        print(mean,std)
        Xtrain = (Xtrain - mean) / std
        Xval = (Xval - mean) / std
        Xtest = (Xtest - mean) / std

    # NOTE: the zero padding is done after the potential whitening
    if border_pad_size > 0:
        Xtrain = zero_pad_border(Xtrain, border_pad_size)
        Xval = zero_pad_border(Xval, border_pad_size)
        Xtest = zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def load_cifar10(data_dir, flatten=False, one_hot=True, normalize_range=False,
        whiten_pixels=True, border_pad_size=0):
    """Loads all of CIFAR-10 in a numpy array.

    Provides a few options for the output formats. For example,
    normalize_range returns the output images with pixel values in [0.0, 1.0].
    The other options are self explanatory. Border padding corresponds to
    upsampling the image by zero padding the border of the image.

    """
    train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
    val_filenames = ['data_batch_5']
    test_filenames = ['test_batch']

    # NOTE: this function uses some arguments from the outer scope, namely
    # flatten, one_hot, normalize_range, and possibly others once added.
    def _load_data(fpath):
        with open(fpath, 'rb') as f:
            try:
                d = cPickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                d = cPickle.load(f, encoding='bytes')
                d = {k.decode(): v for k, v in d.items()}  # change keys into strings

            # for the data
            X = d['data'].astype('float32')

            # reshape the data to the format (num_images, height, width, depth)
            num_images = X.shape[0]
            num_classes = 10
            X = X.reshape( (num_images, 3, 32, 32) )
            X = X.transpose( (0,2,3,1) )
            X = X.astype('float32')

            # transformations based on the argument options.
            if normalize_range:
                X = X / 255.0

            if flatten:
                X = X.reshape( (num_images, -1) )

            # for the labels
            y = np.array(d['labels'])

            if one_hot:
                y_one_hot = np.zeros( (num_images, num_classes), dtype='float32')
                y_one_hot[ np.arange(num_images),  y ] = 1.0
                y = y_one_hot

            return (X, y)

    # NOTE: this function uses some arguments from the outer scope.
    def _load_data_multiple_files(fname_list):

        X_parts = []
        y_parts = []
        for fname in fname_list:
            fpath = os.path.join(data_dir, fname)
            X, y = _load_data(fpath)
            X_parts.append(X)
            y_parts.append(y)

        X_full = np.concatenate(X_parts, axis=0)
        y_full = np.concatenate(y_parts, axis=0)

        return (X_full, y_full)

    Xtrain, ytrain = _load_data_multiple_files(train_filenames)
    Xval, yval = _load_data_multiple_files(val_filenames)
    Xtest, ytest = _load_data_multiple_files(test_filenames)

    if whiten_pixels:
        mean = Xtrain.mean(axis=0)[None, :]
        std = Xtrain.std(axis=0)[None, :]
        Xtrain = (Xtrain - mean) / std
        Xval = (Xval - mean) / std
        Xtest = (Xtest - mean) / std

    # NOTE: the zero padding is done after the potential whitening
    if border_pad_size > 0:
        Xtrain = zero_pad_border(Xtrain, border_pad_size)
        Xval = zero_pad_border(Xval, border_pad_size)
        Xtest = zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images

def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # image shape
    HEIGHT = 96
    WIDTH = 96
    DEPTH = 3

    # size of a single image in bytes
    SIZE = HEIGHT * WIDTH * DEPTH
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image

def download_and_extract(DATA_DIR):
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def load_stl10(data_dir, flatten=False, one_hot=True, normalize_range=False,
        whiten_pixels=True, border_pad_size=0):
    # path to the binary train file with image data
    train_img_path = os.path.join(data_dir,'stl10_binary','train_X.bin')

    # path to the binary train file with labels
    train_label_path = os.path.join(data_dir,'stl10_binary','train_y.bin')

    # path to the binary test file with image data
    test_img_path = os.path.join(data_dir,'stl10_binary','test_X.bin')

    # path to the binary test file with labels
    test_label_path = os.path.join(data_dir,'stl10_binary','test_y.bin')
    
    download_and_extract(data_dir)

    # test to check if the whole dataset is read correctly
    images_train = read_all_images(train_img_path)
    print("Training images",images_train.shape)

    labels_train = read_labels(train_label_path)
    print("Training labels",labels_train.shape)
    
    images_test = read_all_images(test_img_path)
    print("Test images",images_test.shape)

    labels_test = read_labels(test_label_path)
    print("Test labels",labels_test.shape)
    
    Xtrain = images_train.astype(np.float32) / 255.0
    ytrain = labels_train
    for i in range(len(ytrain)) :
        ytrain[i] -= 1
    
    split = int(np.floor(0.9 * Xtrain.shape[0]))
    
    Xval = Xtrain[split:Xtrain.shape[0]]
    yval = ytrain[split:Xtrain.shape[0]]

    Xtrain = Xtrain[:split]
    ytrain = ytrain[:split]

    Xtest, ytest = images_test.astype(np.float32) / 255.0, labels_test
    for i in range(len(ytest)) :
        ytest[i] -= 1
        
    if flatten :
        print("Flatten Not Supported")
    
    if normalize_range :
        print("Normalize Range Not Supported")
    
    if one_hot:
        print("Train Shapes before one hot encoding ",Xtrain.shape, ytrain.shape)
        ytest = idx_to_onehot(ytest, 10).astype(np.float32)
        ytrain = idx_to_onehot(ytrain, 10).astype(np.float32)
        yval = idx_to_onehot(yval, 10).astype(np.float32)
        print("Train Shapes after one hot encoding",Xtrain.shape, ytrain.shape)

    if whiten_pixels:
        mean = Xtrain.mean(axis=0)[None, :]
        std = Xtrain.std(axis=0)[None, :]
        print("Other mean/std", mean.shape, std.shape)
        Xtrain = (Xtrain - mean) / std
        Xval = (Xval - mean) / std
        Xtest = (Xtest - mean) / std

    # NOTE: the zero padding is done after the potential whitening
    if border_pad_size > 0:
        Xtrain = zero_pad_border(Xtrain, border_pad_size)
        Xval = zero_pad_border(Xval, border_pad_size)
        Xtest = zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def load_svhn(data_dir, flatten=False, one_hot=True, normalize_range=False,
        whiten_pixels=True, border_pad_size=0):
    train_path = os.path.join(data_dir, 'train_32x32')
    train_dict = sio.loadmat(train_path)
    X = np.asarray(train_dict['X'])

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train, dtype=np.float32)

    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0

    Xtrain = X_train
    ytrain = np.squeeze(Y_train)
    
    test_path = os.path.join(data_dir, 'test_32x32')
    test_dict = sio.loadmat(test_path)
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test, dtype=np.float32)

    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0

    Xtest = X_test
    ytest = np.squeeze(Y_test)
    
    split = int(np.floor(0.9 * Xtrain.shape[0]))
    
    Xval = Xtrain[split:Xtrain.shape[0]]
    yval = ytrain[split:Xtrain.shape[0]]

    Xtrain = Xtrain[:split]
    ytrain = ytrain[:split]
        
    if flatten :
        print("Flatten Not Supported")
    
    if normalize_range :
        print("Normalize Range Not Supported")
    
    if one_hot:
        print("Train Shapes before one hot encoding ",Xtrain.shape, ytrain.shape)
        ytest = idx_to_onehot(ytest, 10).astype(np.float32)
        ytrain = idx_to_onehot(ytrain, 10).astype(np.float32)
        yval = idx_to_onehot(yval, 10).astype(np.float32)
        print("Train Shapes after one hot encoding",Xtrain.shape, ytrain.shape)

    if whiten_pixels:
        mean = Xtrain.mean(axis=0)[None, :]
        std = Xtrain.std(axis=0)[None, :]
        print("Other mean/std", mean.shape, std.shape)
        Xtrain = (Xtrain - mean) / std
        Xval = (Xval - mean) / std
        Xtest = (Xtest - mean) / std

    # NOTE: the zero padding is done after the potential whitening
    if border_pad_size > 0:
        Xtrain = zero_pad_border(Xtrain, border_pad_size)
        Xval = zero_pad_border(Xval, border_pad_size)
        Xtest = zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def read_devanagari_data(dataset_name, data_directory, class_map, directories_as_labels=True, files='**/*.png'):
    # Create a dataset of file path and class tuples for each file
    filenames = glob.glob(os.path.join(data_directory, files))
    classes = (os.path.basename(os.path.dirname(name)) for name in filenames) if directories_as_labels else [None] * len(filenames)
    dataset = list(zip(filenames, classes))
    num_examples = len(filenames)
    print("Number of examples",num_examples)
    image_set = []
    label_set = []
    for index, sample in enumerate(dataset):
        file_path, label = sample
        image = Image.open(file_path)
        image_raw = np.array(image)
        image_raw = image_raw.reshape(32,32,1)
        image_set.append(image_raw)
        label_set.append(class_map[label])
    image_set = np.asarray(image_set, dtype=np.uint8)
    label_set = np.asarray(label_set, dtype=np.uint8)
    print("Done", dataset_name)
    return image_set, label_set

def load_devanagari(data_dir, flatten=False, one_hot=True, normalize_range=False,
        whiten_pixels=True, border_pad_size=0):
    data_directory = os.path.expanduser(data_dir)
    train_data_dir = os.path.join(data_directory, 'Train')
    test_data_dir = os.path.join(data_directory, 'Test')

    class_names = os.listdir(train_data_dir) # Get names of classes
    class_name2id = { label: index for index, label in enumerate(class_names) } # Map class names to integer labels

    # Persist this mapping so it can be loaded when training for decoding
    with open(os.path.join(data_directory, 'class_name2id.p'), 'wb') as p:
        pickle.dump(class_name2id, p, protocol=pickle.HIGHEST_PROTOCOL)
    
    x_train, y_train = read_devanagari_data('train', train_data_dir, class_name2id)
    combined = list(zip(x_train, y_train))
    random.shuffle(combined)
    
    Xtrain, ytrain = zip(*combined)
    Xtrain = np.asarray(Xtrain, dtype=np.float32)
    ytrain = np.asarray(ytrain, dtype=np.uint8)
    
    x_test, y_test = read_devanagari_data('test', test_data_dir, class_name2id)
    combined = list(zip(x_test, y_test))
    random.shuffle(combined)
    
    Xtest, ytest = zip(*combined)
    Xtest = np.asarray(Xtest, dtype=np.float32)
    ytest = np.asarray(ytest, dtype=np.uint8)

    print(Xtrain.shape, ytrain.shape,Xtrain.dtype, ytrain.dtype)
    print(Xtest.shape, ytest.shape,Xtest.dtype, ytest.dtype)
    
    split = int(np.floor(0.9 * Xtrain.shape[0]))
    
    Xval = Xtrain[split:Xtrain.shape[0]]
    yval = ytrain[split:Xtrain.shape[0]]

    Xtrain = Xtrain[:split]
    ytrain = ytrain[:split]
        
    if flatten :
        print("Flatten Not Supported")
    
    if normalize_range :
        print("Normalize Range Not Supported")
    
    if one_hot:
        print("Train Shapes before one hot encoding ",Xtrain.shape, ytrain.shape)
        ytest = idx_to_onehot(ytest, 46)
        ytrain = idx_to_onehot(ytrain, 46)
        yval = idx_to_onehot(yval, 46)
        print("Train Shapes after one hot encoding",Xtrain.shape, ytrain.shape)

    if whiten_pixels:
        mean = Xtrain.mean()
        std = Xtrain.std()
        print(mean,std)
        Xtrain = (Xtrain - mean) / std
        Xval = (Xval - mean) / std
        Xtest = (Xtest - mean) / std

    # NOTE: the zero padding is done after the potential whitening
    if border_pad_size > 0:
        Xtrain = zero_pad_border(Xtrain, border_pad_size)
        Xval = zero_pad_border(Xval, border_pad_size)
        Xtest = zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def onehot_to_idx(y_onehot):
    y_idx = np.where(y_onehot > 0.0)[1]

    return y_idx

def idx_to_onehot(y_idx, num_classes):
    num_images = y_idx.shape[0]
    y_one_hot = np.zeros( (num_images, num_classes), dtype='float32')
    y_one_hot[ np.arange(num_images),  y_idx ] = 1.0

    return y_one_hot

def center_crop(X, out_height, out_width):
    num_examples, in_height, in_width, in_depth = X.shape
    assert out_height <= in_height and out_width <= in_width

    start_i = (in_height - out_height) // 2
    start_j = (in_width - out_width) // 2
    out_X = X[:, start_i : start_i + out_height, start_j : start_j + out_width, :]

    return out_X

# random crops for each of the images.
def random_crop(X, out_height, out_width):
    num_examples, in_height, in_width, in_depth = X.shape
    # the ouput dimensions have to be smaller or equal that the input dimensions.
    assert out_height <= in_height and out_width <= in_width

    start_is = np.random.randint(in_height - out_height + 1, size=num_examples)
    start_js = np.random.randint(in_width - out_width + 1, size=num_examples)
    out_X = []
    for ind in range(num_examples):
        st_i = start_is[ind]
        st_j = start_js[ind]

        out_Xi = X[ind, st_i : st_i + out_height, st_j : st_j + out_width, :]
        out_X.append(out_Xi)

    out_X = np.array(out_X)
    return out_X

def random_flip_left_right(X, p_flip):
    num_examples, height, width, depth = X.shape

    out_X = X.copy()
    flip_mask = np.random.random(num_examples) < p_flip
    out_X[flip_mask] = out_X[flip_mask, :, ::-1, :]

    return out_X

def per_image_whiten(X):
    """ Subtracts the mean of each image in X and renormalizes them to unit norm.

    """
    num_examples, height, width, depth = X.shape

    X_flat = X.reshape((num_examples, -1))
    X_mean = X_flat.mean(axis=1)
    X_cent = X_flat - X_mean[:, None]
    X_norm = np.sqrt( np.sum( X_cent * X_cent, axis=1) )
    X_out = X_cent / X_norm[:, None]
    X_out = X_out.reshape(X.shape)

    return X_out

# Assumes the following ordering for X: (num_images, height, width, num_channels)
def zero_pad_border(X, pad_size):
    n, height, width, num_channels = X.shape
    X_padded = np.zeros((n, height + 2 * pad_size, width + 2 * pad_size,
        num_channels), dtype='float32')
    X_padded[:, pad_size:height + pad_size, pad_size:width + pad_size, :] = X

    return X_padded

# auxiliary functions for
def get_augment_data_train(out_height, out_width, p_flip):
    def augment_fn(X, y):
        X_out = random_crop(X, out_height, out_width)
        X_out = random_flip_left_right(X_out, p_flip)
        #X_out = per_image_whiten(X_out)
        y_out = y

        return (X_out, y_out)
    return augment_fn

# for evaluation, crop in the middle and do per image "whitening".
def get_augment_data_eval(out_height, out_width):
    def augment_fn(X, y):
        X_out = center_crop(X, out_height, out_width)
        #X_out = per_image_whiten(X_out)
        y_out = y

        return (X_out, y_out)
    return augment_fn


