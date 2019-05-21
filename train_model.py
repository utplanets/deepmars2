#!/usr/bin/env python
"""Convolutional Neural Network Training Functions

Functions for building and training a (UNET) Convolutional Neural Network on
images of the Mars and binary ring targets.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import h5py

from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from deepmars.YNET.model import build_YNET
import deepmars.features.template_match_target as tmt
import click
import logging
from dotenv import find_dotenv, load_dotenv
import os
from tqdm import tqdm
import time
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from imgaug import augmenters as iaa
from joblib import Parallel, delayed
import pickle

# allow free allocation of gpu memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

#previous tuned
#minrad_ = 3
#maxrad_ = 60
#longlat_thresh2_ = 1.8
#rad_thresh_ = 1.0
#template_thresh_ = 0.5
#target_thresh_ = 0.1

#tuning
#minrad_ = 3
#maxrad_ = 60
#longlat_thresh2_ = 1.8
#rad_thresh_ = 1.0
#template_thresh_ = 0.5
#target_thresh_ = 0.05


@click.group()
def dl():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    import sys
    sys.path.append(os.getenv("DM_ROOTDIR"))
    pass


########################
#def get_param_i(param, i):
#    """Gets correct parameter for iteration i.
#
#    Parameters
#    ----------
#    param : list
#        List of model hyperparameters to be iterated over.
#    i : integer
#        Hyperparameter iteration.
#
#    Returns
#    -------
#    Correct hyperparameter for iteration i.
#    """
#    if len(param) > i:
#        return param[i]
#    else:
#        return param[0]

########################


#def custom_image_generator(data, target, batch_size=32):
#    """Custom image generator that manipulates image/target pairs to prevent
#    overfitting in the Convolutional Neural Network.
#
#    Parameters
#    ----------
#    data : array
#        Input images.
#    target : array
#        Target images.
#    batch_size : int, optional
#        Batch size for image manipulation.
#
#    Yields
#    ------
#    Manipulated images and targets.
#    """
#    
#    D = data.shape[0]
#    while True:
#        shuffle_index = np.arange(D)
#        # only shuffle once each loop through the data
#        np.random.shuffle(shuffle_index)
#        for i in np.arange(0, len(data), batch_size):
#            index = shuffle_index[i:i + batch_size]
#            d, t = data[index].copy(), target[index].copy()
#            
#            d = d.transpose([1,0,2,3,4])
#            yield ([d[0],d[1]], t)
#            del d
#            del t


def generator_with_replacement(data, target, batch_size=32):
    augmentation = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.OneOf([iaa.Affine(rotate=0), iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180), iaa.Affine(rotate=270)])])
    D = data.shape[0]
    while True:
        indices = [np.random.randint(D) for _ in range(batch_size)]
        d, t = data[indices].copy(), target[indices].copy()
        #d = d.transpose([1,0,2,3,4])
        
        # -------
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
        # --------
        
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

    w = np.where(N_match == 0)

    w = np.where(N_match > 0)
    counter, N_match, N_csv, N_detect,\
        mrad, err_lo, err_la, errr_, frac_dupes =\
        counter[w], N_match[w], N_csv[w], N_detect[w],\
        mrad[w], err_lo[w], err_la[w], err_r[w], frac_duplicates[w]

    precision = N_match / (N_match + (N_detect - N_match))
    recall = N_match / N_csv
    fscore = (1 + beta**2) * (recall * precision) / \
             (precision * beta**2 + recall)
    diff = N_detect - N_match
    frac_new = diff / (N_detect + diff)
    frac_new2 = diff / (N_csv + diff)
    frac_duplicates = frac_dupes

    return dict(precision=precision,
                recall=recall,
                fscore=fscore,
                frac_new=frac_new,
                frac_new2=frac_new2,
                err_lo=err_lo,
                err_la=err_la,
                err_r=err_r,
                frac_duplicates=frac_duplicates,
                maxrad=mrad,
                counter=counter, N_match=N_match, N_csv=N_csv)


def get_metrics(data, craters_images, model, name, minrad, maxrad,
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
    X = data[0]
    Y = data[1]
    craters, images = craters_images
    # Get csvs of human-counted craters
    csvs = []
#    minrad, maxrad = 3, 50
    cutrad, n_csvs = 0.8, len(X)
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
        # remove small/large/half craters
#        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
#        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
#        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
#        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
#        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x (pix)'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y (pix)'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x (pix)'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y (pix)'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 3:    # Exclude csvs with few craters
            csvs.append([-1])
        else:
#            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csv_coords = np.asarray((csv['x (pix)'], csv['y (pix)'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    # Calculate custom metrics
#    print("csvs: {}".format(len(csvs)))
#    print("")
#    print("*********Custom Loss*********")
#    recall, precision, fscore = [], [], []
#    frac_new, frac_new2, mrad = [], [], []
#    err_lo, err_la, err_r = [], [], []
#    frac_duplicates = []

    if isinstance(model, Model):
        preds = None
#        print(X[6].min(),X[6].max(),X.dtype,np.percentile(X[6],99))
        X = X.transpose([1,0,2,3,4])
        preds = model.predict([X[0], X[1]], verbose=1, batch_size=5)
        # save
        h5f = h5py.File("predictions.hdf5", 'w')
        h5f.create_dataset(name, data=preds)
        print("Successfully generated and saved model predictions.")
    elif model is None:
        h5f = h5py.File("predictions.hdf5", 'r')
        preds = h5f[name][:]
    else:
        preds = model
    # print(csvs)
    countme = [i for i in range(n_csvs) if len(csvs[i]) >= 3]
#    print("Processing {} fields".format(len(countme)))

    # preds contains a large number of predictions,
    # so we run the template code in parallel.
    res = Parallel(n_jobs=16,
                   verbose=0)(delayed(t2c)(preds[i][:,:,0], csvs[i], i,
                                           minrad=minrad,
                                           maxrad=maxrad,
                                           longlat_thresh2=longlat_thresh2,
                                           rad_thresh=rad_thresh,
                                           template_thresh=template_thresh,
                                           target_thresh=target_thresh)
                              for i in range(n_csvs) if len(csvs[i]) >= 3)

    if len(res) == 0:
        print("No valid results: ", res)
        return None
    # At this point we've processed the predictions with the template matching
    # algorithm, now calculate the metrics from the data.
    diag = diagnostic(res, beta)
    return diag
#    print(len(diag["recall"]))
#    # print("binary XE score = %f" % model.evaluate(X, Y))
#    if len(diag["recall"]) > 3:
#        metric_data = [("N_match/N_csv (recall)", diag["recall"]),
#                       ("N_match/(N_match + (N_detect-N_match)) (precision)",
#                       diag["precision"]),
#                       ("F_{} score".format(beta), diag["fscore"]),
#                       ("(N_detect - N_match)/N_detect" +
#                        "(fraction of craters that are new)",
#                       diag["frac_new"]),
#                       ("(N_detect - N_match)/N_csv (fraction" +
#                        "of craters that are new, 2)", diag["frac_new2"])]
#
#        for fname, data in metric_data:
#            print("mean and std of %s = %f, %f" %
#                  (fname, np.mean(data), np.std(data)))
#        for fname, data in [("fractional longitude diff", diag["err_lo"]),
#                            ("fractional latitude diff", diag["err_la"]),
#                            ("fractional radius diff", diag["err_r"]),
#                            ]:
#            print("median and IQR %s = %f, 25:%f, 75:%f" %
#                  (fname,
#                   np.median(data),
#                   np.percentile(data, 25),
#                   np.percentile(data, 75)))
#
#        print("""mean and std of maximum detected pixel radius in an image =
#             %f, %f""" % (np.mean(diag["maxrad"]), np.std(diag["maxrad"])))
#        print("""absolute maximum detected pixel radius over all images =
#              %f""" % np.max(diag["maxrad"]))
#        print("")


########################


def test_model(Data, Craters, MP, minrad, maxrad, longlat_thresh2, rad_thresh,
               template_thresh, target_thresh):
    # Static params
#    dim = MP['dim']
#    nb_epoch = MP['epochs']
#    bs = MP['bs']
#
#    # Iterating params
#    FL = MP['filter_length'][0]
#    learn_rate = MP['lr'][0]
#    n_filters = MP['n_filters'][0]
#    init = MP['init'][0]
#    lmbda = MP['lambda'][0]
#    drop = MP['dropout'][0]

    try:
        model = load_model(MP["model"])
    except:
        model = None

    diag = get_metrics(Data[MP["test_dataset"]],
                       Craters[MP["test_dataset"]], model, MP["test_dataset"],
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
    save_folder = os.path.join(os.getenv('DM_ROOTDIR'), 'YNET/models', now)
    os.mkdir(save_folder)
    save_name = save_folder + '/{epoch:02d}-{val_loss:.2f}.hdf5'
    save_model = ModelCheckpoint(os.path.join(os.getenv('DM_ROOTDIR'),
                                                       save_name))
    log_dir=os.path.join(os.getenv('DM_ROOTDIR'), 'YNET/logs', now)
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

#    print("###################################")
#    print("##########END_OF_RUN_INFO##########")
#    print("""learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d
#          n_train=%d, img_dimensions=%d, init=%s, n_filters=%d, lambda=%e
#          dropout=%f""" % (learn_rate, bs, FL, nb_epoch, MP['n_train'],
#                           MP['dim'], init, n_filters, lmbda, drop))
#    if MP["calculate_custom_loss"]:
#        get_metrics(Data['test'], Craters['test'], dim, model, "test")
#    print("###################################")
#    print("###################################")

########################


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
#        minrad_ = 3
#        maxrad_ = 60
        longlat_thresh2_ = 1.8
        rad_thresh_ = 1.0
#        template_thresh_ = 0.65
#        target_thresh_ = 0.25
        
        grid = {}
        minradii = [3,8,13,18,23,28,33,38,43,48,53,58]
        template_threshes = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        target_threshes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        for minrad_ in minradii:
            maxrad_ = minrad_ + 5
            for template_thresh_ in template_threshes:
                for target_thresh_ in target_threshes:
                    
                    diag = test_model(Data, Craters, MP, minrad=minrad_,
                                      maxrad=maxrad_, 
                                      longlat_thresh2=longlat_thresh2_,
                                      rad_thresh=rad_thresh_,
                                      template_thresh=template_thresh_,
                                      target_thresh=target_thresh_)
                    grid[(minrad_, template_thresh_, target_thresh_)] = diag
                    print('------------------------------------------------')
                    print('radius: {} - {}'.format(minrad_, maxrad_))
                    print('template threshold: {}'.format(template_thresh_))
                    print('target threshold: {}'.format(target_thresh_))
                    try:
                        print('Precision: {}'.format(diag['precision'].mean()))
                        print('Recall: {}'.format(diag['recall'].mean()))
                        print('F1: {}'.format(diag['fscore'].mean()))
                    except:
                        print('No Craters Found')
                    pickle.dump(grid, open('grid2.pkl', 'wb'))
    else:
        train_and_test_model(Data, Craters, MP)


@dl.command()
@click.option("--test", is_flag=True, default=False)
@click.option("--test_dataset", default="test")
@click.option("--model", default=None)
def train_model(test, test_dataset, model):
    """Run Convolutional Neural Network Training

    Execute the training of a (UNET) Convolutional Neural Network on
    images of the Moon and binary ring targets.
    """

    # Model Parameters
    MP = {}

    # Directory of train/dev/test image and crater hdf5 files.
    MP['dir'] = os.path.join(os.getenv("DM_ROOTDIR"), 'data/processed/')

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

    MP['n_train'] = len(MP["train_indices"]) * 1000
    MP['n_dev'] = len(MP["dev_indices"]) * 1000
    MP['n_test'] = len(MP["test_indices"]) * 1000

    # Save model (binary flag) and directory.
    #MP['save_models'] = 1
    #MP["calculate_custom_loss"] = False
    #MP['save_dir'] = 'models'
    #MP['final_save_name'] = 'model.h5'

    # initial model filename
    MP["model"] = model

    # testing only
    MP["test"] = test
    MP["test_dataset"] = test_dataset
#
#    # Model Parameters (to potentially iterate over, keep in lists).
#    # runs.csv looks like
#    # filter_length,lr,n_filters,init,lambda,dropout
#    # 3,0.0001,112,he_normal,1e-6,0.15
#    #
#    # each line is a new run.
#    df = pd.read_csv("runs.csv")
#    for na, ty in [("filter_length", int),
#                   ("lr", float),
#                   ("n_filters", int),
#                   ("init", str),
#                   ("lambda", float),
#                   ("dropout", float)]:
#        MP[na] = df[na].astype(ty).values
#
#    MP['N_runs'] = len(MP['lambda'])    # Number of runs
    MP['filter_length'] = [3]           # Filter length
    MP['lr'] = [0.0001]                # Learning rate
    MP['n_filters'] = [112]             # Number of filters
    MP['init'] = ['he_normal']          # Weight initialization
    MP['lambda'] = [1e-6]               # Weight regularization
    MP['dropout'] = [0.15]              # Dropout fraction
    
    MP['prefix'] = 'ran2'
    
    print(MP)
    get_models(MP)


if __name__ == '__main__':
    dl()
