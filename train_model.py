#!/usr/bin/env python
"""Convolutional Neural Network Training Functions

Functions for building and training a (UNET) Convolutional Neural Network on
images of the Mars and binary ring targets.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import h5py

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from deepmars.YNET.model import build_YNET
import deepmars.features.template_match_target as tmt
import click
import logging
#from dotenv import find_dotenv, load_dotenv
import os
from tqdm import tqdm
import time
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from imgaug import augmenters as iaa
from joblib import Parallel, delayed
import config as cfg
from cratertools.metric import match_craters

# allow free allocation of gpu memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


@click.group()
def dl():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


def generator_with_replacement(data, target, batch_size=32):
    augmentation = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.OneOf([iaa.Affine(rotate=0), iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180), iaa.Affine(rotate=270)])])
    D = data.shape[0]
    while True:
        indices = [np.random.randint(D) for _ in range(batch_size)]
        d, t = data[indices].copy(), target[indices].copy()
        
        DEM = d[:,0,...]
        IR = d[:,1,...]
        DEM = np.squeeze(DEM.transpose([3,1,2,0]))
        IR = np.squeeze(IR.transpose([3,1,2,0]))
        t = np.squeeze(t.transpose([3,1,2,0]))
        
        det = augmentation.to_deterministic()
        
        DEM = det.augment_image(DEM)
        IR = det.augment_image(IR)
        t = det.augment_image(t)
        
        DEM = DEM.transpose([2,0,1])[..., np.newaxis]
        IR = IR.transpose([2,0,1])[..., np.newaxis]
        t = t.transpose([2,0,1])[..., np.newaxis]
        
        yield ([DEM,IR], t)
        del d
        del t


def t2c(pred, csv, i, minrad, maxrad, longlat_thresh2, rad_thresh,
        template_thresh, target_thresh):
    return np.hstack([i,
                      tmt.template_match_t2c(pred, csv,
                                             minrad=minrad,
                                             maxrad=maxrad,
                                             longlat_thresh2=longlat_thresh2,
                                             rad_thresh=rad_thresh,
                                             template_thresh=template_thresh,
                                             target_thresh=target_thresh)][1:])


def diagnostic(res, beta):
    """Calculate the metrics from the predictions compared to the CSV.

    Parameters
    ------------
    res: list of results containing:
        image number, number of matched, number of existing craters, number of
        detected craters, maximum radius detected, mean error in longitude,
        mean error in latitude, mean error in radius, fraction of duplicates
        in detections.
    beta : int
        Beta value when calculating F-beta score.

    Returns
    -------
    dictionary : metrics stored in a dictionary
    """

    counter, N_match, N_csv, N_detect,\
        mrad, err_lo, err_la, err_r, frac_duplicates = np.array(res).T
    try:
        positives = N_detect.sum()
        true_positives = N_match.sum()
        ground_truth = N_csv.sum()
        precision = true_positives / positives
        recall = true_positives / ground_truth
        fscore = (1 + beta**2) * (recall * precision) / \
                 (precision * beta**2 + recall)
    except:
        # division by zero occured, so no craters were detected
        precision = 0
        recall = 0
        fscore = 0
    
    return dict(precision=precision, recall=recall, fscore=fscore)
    

def get_metrics(data, craters_images, model, preds, name, minrad, maxrad,
                longlat_thresh2, rad_thresh, template_thresh, target_thresh,
                dim=256, beta=1, offset=0, rmv_oor_csvs=0):
    """Function that prints pertinent metrics at the end of each epoch.

    Parameters
    ----------
    data : hdf5
        Input images.
    craters : hdf5
        Pandas arrays of human-counted crater data.
    model : keras model object
        Keras model
    beta : int, optional
        Beta value when calculating F-beta score. Defaults to 1.
    """
    
    # Either a model is provided to make predictions or the predictions have
    # already been made
    assert (model is not None) or (preds is not None)
    
    X = data[0]
    craters, images = craters_images
    # Get csvs of human-counted craters
    csvs = []
    cutrad = 0.8
    n_csvs = len(X)
    diam = 'Diameter (pix)'

    for i in range(len(X)):
        imname = images[i]  # name = "img_{0:05d}".format(i)
        found = False
        for crat in craters:
            if imname in crat:
                csv = crat[imname]
                found = True
        if not found:
            csvs.append([-2])
            continue
        # Remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x (pix)'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y (pix)'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x (pix)'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y (pix)'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 3:    # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x (pix)'], csv['y (pix)'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    if model is not None:
        # A model has been provided so we must make predictions
        X = X.transpose([1,0,2,3,4])
        preds = model.predict([X[0], X[1]], verbose=1, batch_size=5)
        # Save predictions
        h5f = h5py.File('predictions.hdf5', 'w')
        h5f.create_dataset(name, data=preds)
        print('Successfully generated and saved model predictions.')

    # preds contains a large number of predictions,
    # so we run the template code in parallel.
    res = Parallel(n_jobs=16,
                   verbose=1)(delayed(t2c)(preds[i][:,:,0], csvs[i], i,
                                           minrad=minrad,
                                           maxrad=maxrad,
                                           longlat_thresh2=longlat_thresh2,
                                           rad_thresh=rad_thresh,
                                           template_thresh=template_thresh,
                                           target_thresh=target_thresh)
                              for i in range(n_csvs) if len(csvs[i]) >= 3)

    if len(res) == 0:
        print('No valid results: ', res)
        return None
    
    # At this point we've processed the predictions with the template matching
    # algorithm, now calculate the metrics from the data.
    
    return diagnostic(res, beta)


def test_model(Data, Craters, MP, minrad, maxrad, longlat_thresh2, rad_thresh,
               template_thresh, target_thresh):
    
    if MP['model'] is not None:
        model = load_model(MP['model'])
        preds = None
    else:
        # no model specified so load predictions file
        h5f = h5py.File(MP['predictions'], 'r')
        preds = h5f[MP['test_dataset']][:]
        model = None

    diag = get_metrics(Data[MP["test_dataset"]],
                       Craters[MP["test_dataset"]],
                       model,
                       preds,
                       MP["test_dataset"],
                       minrad=minrad, maxrad=maxrad,
                       longlat_thresh2=longlat_thresh2,
                       rad_thresh=rad_thresh, template_thresh=template_thresh,
                       target_thresh=target_thresh)
    
    return diag


def train_and_test_model(Data, Craters, MP):
    """Function that trains, tests and saves the model, printing out metrics
    after each model.

    Parameters
    ----------
    Data : dict
        Inputs and Target Moon data.
    Craters : dict
        Human-counted crater data.
    MP : dict
        Contains all relevant parameters.
    i_MP : int
        Iteration number (when iterating over hypers).
    """
    
    # Parameters
    dim = MP['dim']
    nb_epoch = MP['epochs']
    bs = MP['bs']
    FL = MP['filter_length'][0]
    learn_rate = MP['lr'][0]
    n_filters = MP['n_filters'][0]
#    init = MP['init']
    lmbda = MP['lambda'][0]
    drop = MP['dropout'][0]

    # Build model
    if MP["model"] is not None:
        model = load_model(MP["model"])
    else:
        model = build_YNET(4, 2, dim, FL, n_filters, drop, learn_rate,
                           lmbda=lmbda, activation_function='relu')

    
    # Callbacks
    now = time.strftime('%c')
    n_samples = 1000
    save_folder = os.path.join(cfg.input_root, 'YNET/models', now)
    os.mkdir(save_folder)
    save_name = save_folder + '/{epoch:02d}-{val_loss:.2f}.hdf5'
    save_model = ModelCheckpoint(os.path.join(cfg.input_root,
                                                       save_name))
    log_dir=os.path.join(cfg.input_root, 'YNET/logs', now)
    tensorboard = TensorBoard(log_dir, batch_size=bs, histogram_freq=1) 
    tbi_test = TensorBoardImage(log_dir)
    
    val_sample = slice(0,10)
    model.fit_generator(generator_with_replacement(Data['train'][0],
                                                   Data['train'][1],
                                                   batch_size=bs),
                        steps_per_epoch=n_samples // bs,
                        verbose=1,
                        validation_data=([Data['dev'][0][val_sample,0], Data['dev'][0][val_sample,1]], Data['dev'][1][val_sample]),
                        callbacks=[save_model, tensorboard, tbi_test],
                        epochs=nb_epoch)


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width = tensor.shape
    tensor = (255*tensor).astype('uint8')
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=1,
                         encoded_image_string=image_string)


class TensorBoardImage(Callback):
    def __init__(self, log_dir):
        super().__init__() 
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs={}):
        data = self.validation_data
        input_DEM = data[0]
        input_IR = data[1]
        target_masks = data[2]
        print('gerenating validation prediction images')
        preds = self.model.predict([input_DEM, input_IR], batch_size=1)
        
        images = []
        for i in tqdm(range(input_DEM.shape[0])):
            img = np.vstack([np.hstack([input_DEM[i,:,:,0],
                                        input_IR[i,:,:,0]]),
                             np.hstack([target_masks[i,:,:,0],
                                        preds[i,:,:,0]])
                            ])
        
            images.append(make_image(img))
        
        summary = tf.Summary(value=[
                tf.Summary.Value(tag='Image {:02d}'.format(i), image=images[i])
                for i in range(len(images))])
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(summary, epoch)
        writer.close()

        return
        


def get_models(MP):
    """Top-level function that loads data files and calls train_and_test_model.

    Parameters
    ----------
    MP : dict
        Model Parameters.
    """
    dir = MP['dir']
    n_train, n_dev, n_test = MP['n_train'], MP['n_dev'], MP['n_test']
    prefix = MP['prefix']

    # Load data
    def load_files(numbers, test, this_dataset):
        res0 = []
        res1 = []
        files = []
        craters = []
        images = []
        npic = 0
        if not test or (test and this_dataset):
            for n in tqdm(numbers):
                files.append(h5py.File(os.path.join(
                    dir, prefix + "_images_{0:05d}.hdf5".format(n)), 'r'))
                images.extend(["img_{0:05d}".format(a)
                              for a in np.arange(n, n + 1000)])
                #res0.append(files[-1]["input_DEM"][:].astype('float32'))
                             
                res0.append([files[-1]["input_DEM"][:].astype('float32'),
                             files[-1]["input_IR"][:].astype('float32')])
                npic = npic + len(res0[-1])
                res1.append(files[-1]["target_masks"][:].astype('float32'))
                files[-1].close()
                craters.append(pd.HDFStore(os.path.join(
                    dir, prefix + "_craters_{0:05d}.hdf5".format(n)), 'r'))
            res0 = np.vstack(res0).transpose([1,0,2,3])
            res1 = np.vstack(res1)
        return files, res0, res1, npic, craters, images

    train_files,\
        train0,\
        train1,\
        Ntrain,\
        train_craters,\
        train_images = load_files(MP["train_indices"],
                                  MP["test"],
                                  MP["test_dataset"] == "train")
    print(Ntrain, n_train)

    dev_files,\
        dev0,\
        dev1,\
        Ndev,\
        dev_craters,\
        dev_images = load_files(MP["dev_indices"],
                                MP["test"],
                                MP["test_dataset"] == "dev")
    print(Ndev, n_dev)

    test_files,\
        test0,\
        test1,\
        Ntest,\
        test_craters,\
        test_images = load_files(MP["test_indices"],
                                 MP["test"],
                                 MP["test_dataset"] == "test")
    print(Ntest, n_test)

    Data = {
        "train": [train0, train1],
        "dev": [dev0, dev1],
        "test": [test0[:n_test], test1[:n_test]]
        }

    def preprocess(Data):
        for key in Data:
            if len(Data[key][0]) == 0:
                continue
            for i in range(len(Data[key])):
                newdim = list(Data[key][i].shape) + [1]
                Data[key][i] = Data[key][i].reshape(*newdim)
           
    preprocess(Data)

    # Load ground-truth craters
    Craters = {
        'train': [train_craters, train_images],
        'dev': [dev_craters, dev_images],
        'test': [test_craters, test_images]
    }

#    # Iterate over parameters
    if MP["test"]:
        minrad_ = 3
        maxrad_ = 60
        longlat_thresh2_ = 1.8
        rad_thresh_ = 1.0
        template_thresh_ = 0.5
        target_thresh_ = 0.1
        
        diag = test_model(Data, Craters, MP, minrad=minrad_,
                          maxrad=maxrad_, 
                          longlat_thresh2=longlat_thresh2_,
                          rad_thresh=rad_thresh_,
                          template_thresh=template_thresh_,
                          target_thresh=target_thresh_)
        print('Precision: {}'.format(diag['precision']))
        print('Recall: {}'.format(diag['recall']))
        print('F1: {}'.format(diag['fscore']))
    else:
        train_and_test_model(Data, Craters, MP)


@dl.command()
@click.option("--test", is_flag=True, default=False)
@click.option("--test_dataset", default="test")
@click.option("--model", default=None)
@click.option("--predictions", default=None)
def train_model(test, test_dataset, model, predictions):
    """Run Convolutional Neural Network Training

    Execute the training of a (UNET) Convolutional Neural Network on
    images of the Moon and binary ring targets.
    """

    # Model Parameters
    MP = {}

    # Directory of train/dev/test image and crater hdf5 files.
    MP['dir'] = os.path.join(cfg.input_root, 'data/processed/')

    # Image width/height, assuming square images.
    MP['dim'] = 256

    # Batch size: smaller values = less memory, less accurate gradient estimate
    MP['bs'] = 7

    # Number of training epochs.
    MP['epochs'] = 800
    
    # Number of samples per epoch
    MP['per_epoch'] = 1000

    MP['train_indices'] = list(np.arange(0, 40000, 1000))
    MP['dev_indices'] = [41000]#list(np.arange(161000, 206000, 4000))
    MP['test_indices'] = [42000]#list(np.arange(163000, 206000, 4000))

    MP['n_train'] = len(MP['train_indices']) * 1000
    MP['n_dev'] = len(MP['dev_indices']) * 1000
    MP['n_test'] = len(MP['test_indices']) * 1000

    # initial model filename
    MP['model'] = model
    
    # initial predictions filename
    MP['predictions'] = predictions

    # testing only
    MP["test"] = test
    MP["test_dataset"] = test_dataset

    MP['filter_length'] = [3]           # Filter length
    MP['lr'] = [0.0001]                 # Learning rate
    MP['n_filters'] = [112]             # Number of filters
    MP['init'] = ['he_normal']          # Weight initialization
    MP['lambda'] = [1e-6]               # Weight regularization
    MP['dropout'] = [0.15]              # Dropout fraction
    
    MP['prefix'] = 'ran2'
    
    print(MP)
    get_models(MP)


if __name__ == '__main__':
    dl()
