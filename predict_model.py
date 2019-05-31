#!/usr/bin/env python
"""Unique Crater Distribution Functions

Functions for extracting craters from model target predictions and filtering
out duplicates.
"""
import click
import logging
import keras.models as km
import deepmars2.features.template_match_target as tmt
import numpy as np
import h5py
import os
import time
import pandas as pd
from joblib import Parallel, delayed
import config as cfg
from tqdm import tqdm
from pyproj import Transformer
from cratertools import metric
from deepmars2.YNET.model import weighted_cross_entropy
from deepmars2.ResUNET.model import dice_loss

# Reduce Tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_model(model=None):
    if isinstance(model, str):
#        model = km.load_model(model, custom_objects={'weighted_cross_entropy':
#            weighted_cross_entropy})
        model = km.load_model(model, custom_objects={'dice_loss': dice_loss})
    return model


def get_model_preds(CP):
    """Reads in or generates model predictions.

    Parameters
    ----------
    CP : dict
        Containins directory locations for loading data and storing
        predictions.

    Returns
    -------
    craters : h5py
        Model predictions.
    """
    logger = logging.getLogger(__name__)

    n_imgs, dtype = CP["n_imgs"], CP["datatype"]
    logger.info("Reading %s" % CP["dir_data"])

    data = h5py.File(CP["dir_data"], "r")
    if n_imgs < 0:
        n_imgs = data["input_DEM"].shape[0]

    Data = {
        dtype: [ 
            data["input_DEM"][:n_imgs].astype("float32"),
            data["input_IR"][:n_imgs].astype("float32"),
            data["target_masks"][:n_imgs].astype("float32"),
        ]
    }
    data.close()
    
    def preprocess(Data):
        for key in Data:
            if len(Data[key][0]) == 0:
                continue
            for i in range(len(Data[key])):
                newdim = list(Data[key][i].shape) + [1]
                Data[key][i] = Data[key][i].reshape(*newdim)

    preprocess(Data)

    model = load_model(CP["dir_model"])
    logger.info("Making prediction on %d images" % n_imgs)
    #preds = model.predict([Data[dtype][0], Data[dtype][1]], batch_size=10)
    preds = model.predict(Data[dtype][0], batch_size=10)
    logger.info("Finished prediction on %d images" % n_imgs)
    # save
    h5f = h5py.File(CP["dir_preds"], "w")
    h5f.create_dataset(dtype, data=preds, compression="gzip", compression_opts=9)
    print("Successfully generated and saved model predictions.")
    return preds


#########################


def add_unique_craters(craters, craters_unique, thresh_longlat2, thresh_rad):
    """Generates unique crater distribution by filtering out duplicates.

    Parameters
    ----------
    craters : array
        Crater tuples from a single image in the form (long, lat, radius).
    craters_unique : array
        Master array of unique crater tuples in the form (long, lat, radius)
    thresh_longlat2 : float.
        Hyperparameter that controls the minimum squared longitude/latitude
        difference between craters to be considered unique entries.
    thresh_rad : float
        Hyperparaeter that controls the minimum squared radius difference
        between craters to be considered unique entries.

    Returns
    -------
    craters_unique : array
        Modified master array of unique crater tuples with new crater entries.
    """
    k2d = 180.0 / (np.pi * 3389.0)  # km to deg
    Long, Lat, Rad = craters_unique.T
    for j in range(len(craters)):
        lo, la, r = craters[j].T
        la_m = (la + Lat) / 2.0
        minr = np.minimum(r, Rad)  # be liberal when filtering dupes

        # duplicate filtering criteria
        dL = ((Long - lo) / (minr * k2d / np.cos(np.pi * la_m / 180.0))) ** 2 + (
            (Lat - la) / (minr * k2d)
        ) ** 2
        dR = np.abs(Rad - r) / minr
        index = (dR < thresh_rad) & (dL < thresh_longlat2)

        if len(np.where(index)[0]) == 0:
            craters_unique = np.vstack((craters_unique, craters[j]))
    return craters_unique


def match_template(pred, craters, i, index, dim, withmatches=False):
    img = 'img_{:05d}'.format(index + i)
    valid = False
    diam = "Diameter (pix)"
    if withmatches:
        N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, frac_dupes = (
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        )

        if img in craters:
            csv = craters[img]
            found = True
        if found:
            minrad, maxrad = 3, 50
            #cutrad = 0.8
            cutrad = 0
            csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
            csv = csv[(csv["x"] + cutrad * csv[diam] / 2 <= dim[0])]
            csv = csv[(csv["y"] + cutrad * csv[diam] / 2 <= dim[1])]
            csv = csv[(csv["x"] - cutrad * csv[diam] / 2 > 0)]
            csv = csv[(csv["y"] - cutrad * csv[diam] / 2 > 0)]
            if len(csv) >= 3:
                valid = True
                csv = np.asarray((csv["x"], csv["y"], csv[diam] / 2)).T

    if valid:
        coords, N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, frac_dupes = tmt.template_match_t2c(
            pred, csv
        )
        df2 = pd.DataFrame(
            np.array(
                [N_match, N_csv, N_detect, maxr, err_lo, err_la, err_r, frac_dupes]
            )[None, :],
            columns=[
                "N_match",
                "N_csv",
                "N_detect",
                "maxr",
                "err_lo",
                "err_la",
                "err_r",
                "frac_dupes",
            ],
            index=[img],
        )
    else:
        coords = tmt.template_match_t(pred)
        df2 = None
    return [coords, df2]


def long_lat_rad_km_from_pix(coords, box_size, central_lat_lon, dim=256):
    assert coords.min() >= 0 and coords.max() <= 256
    x = coords[:,0].astype('float64') - dim//2
    y = coords[:,1].astype('float64') - dim//2
    r = coords[:,2].astype('float64')
    lat_0, lon_0 = central_lat_lon
    deg_per_pix = box_size / dim
    x *= deg_per_pix
    y *= deg_per_pix
    r *= deg_per_pix
    pipeline_str = (
        "proj=pipeline "
        "step proj=unitconvert xy_in=deg xy_out=rad "
        "step proj=eqc "
        "step proj=ortho inv lat_0={} lon_0={} "
        "step proj=unitconvert xy_in=rad xy_out=deg"
    ).format(lat_0, lon_0)
    transformer = Transformer.from_pipeline(pipeline_str)
    lon, lat = transformer.transform(y, x)
    assert (-180 <= lon).all() and (lon <= 180).all()
    assert (-90 <= lat).all() and (lat <= 90).all()
    km_per_deg = 2 * np.pi * 3389.5 / 360
    r *= km_per_deg
    return np.vstack([lon, lat, r]).T
    


def extract_unique_craters(
    CP, craters_unique, index=0, start=0, stop=-1, withmatches=True
):
    """Top level function that extracts craters from model predictions,
    converts craters from pixel to real (degree, km) coordinates, and filters
    out duplicate detections across images.

    Parameters
    ----------
    CP : dict
        Crater Parameters needed to run the code.
    craters_unique : array
        Empty master array of unique crater tuples in the form
        (long, lat, radius).

    Returns
    -------
    craters_unique : array
        Filled master array of unique crater tuples.
    """
    logger = logging.getLogger(__name__)
    # Load/generate model preds
    try:
        preds = h5py.File(CP["dir_preds"], "r")[CP["datatype"]]
        logger.info("Loaded model predictions successfully")
    except:
        logger.info("Couldnt load model predictions, generating")
        preds = get_model_preds(CP)

    # need for long/lat bounds
    P = h5py.File(CP["dir_data"], 'r')

    dim = (float(CP["dim"]), float(CP["dim"]))

    N_matches_tot = 0
    if start < 0:
        start = 0
    if stop < 0:
        stop = P["input_DEM"].shape[0]

    start = np.clip(start, 0, P["input_DEM"].shape[0] - 1)
    stop = np.clip(stop, 1, P["input_DEM"].shape[0])
    craters_h5 = pd.HDFStore(CP["dir_craters"], "w")

#    csvs = []
    if withmatches:
        craters = pd.HDFStore(CP["dir_input_craters"], "r")
        matches = []

    full_craters = dict()
    if withmatches:
        for i in range(start, stop):
            img = 'img_{:05d}'.format(index + i)
            if img in craters:
                full_craters[img] = craters[img]
    res = Parallel(n_jobs=16, verbose=0)(
        delayed(match_template)(
            preds[i], full_craters, i, index, dim, withmatches=withmatches
        )
        for i in tqdm(range(start, stop))
    )

    for i in range(start, stop):
        coords, df2 = res[i]
        if withmatches:
            matches.append(df2)
        img = 'img_{:05d}'.format(index + i)
        
        # convert, add to master dist
        if len(coords) > 0:
            new_craters_unique = long_lat_rad_km_from_pix(coords, P['box_size'][i], P['central_lat_lon'][i])
            N_matches_tot += len(coords)

            # Only add unique (non-duplicate) craters
            if len(craters_unique) > 0:
                craters_unique = add_unique_craters(
                    new_craters_unique, craters_unique, CP["llt2"], CP["rt"]
                )
            else:
                craters_unique = np.concatenate((craters_unique, new_craters_unique))
            data = np.hstack(
                [
                    new_craters_unique * np.array([1, 1, 2])[None, :],
                    coords * np.array([1, 1, 2])[None, :],
                ]
            )
            df = pd.DataFrame(
                data,
                columns=["Long", "Lat", "Diameter (km)", "x", "y", "Diameter (pix)"],
            )
            craters_h5[img] = df[
                ["Lat", "Long", "Diameter (km)", "x", "y", "Diameter (pix)"]
            ]
            craters_h5.flush()

    logger.info(
        "Saving to %s with %d 6 craters" % (CP["dir_result"], len(craters_unique))
    )
    np.save(CP["dir_result"], craters_unique)
    alldata = craters_unique * np.array([1, 1, 2])[None, :]
    df = pd.DataFrame(alldata, columns=["Long", "Lat", "Diameter (km)"])
    craters_h5["all"] = df[["Lat", "Long", "Diameter (km)"]]
    if withmatches:
        craters_h5["matches"] = pd.concat(matches)
        craters.close()
    craters_h5.flush()
    craters_h5.close()

    return craters_unique


@click.group()
def predict():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    pass


@predict.command()
@click.option("--index", type=int, default=None)
@click.option("--prefix", default="ran2")
@click.option("--output_prefix", default=None)
@click.option("--model", default='/disks/work/james/deepmars2/ResUNET/models/Thu May 30 17:49:54 2019/99-0.56.hdf5')
def cnn_prediction(index, prefix, output_prefix, model):
    """ CNN predictions.

    Run the CNN on a file and generate the output file but do not
    process the file with the template matching code.

    """

    logger = logging.getLogger(__name__)
    logger.info("making predictions.")
    start_time = time.time()

    indexstr = "_{:05d}".format(index)

    if output_prefix is None:
        output_prefix = prefix
    # Crater Parameters
    CP = dict(
        dim=256,
        datatype=prefix,
        n_imgs=-1,
        dir_model=model,
        dir_data=os.path.join(
            cfg.root_dir,
            "data/processed/%s_images%s.hdf5" % (prefix, indexstr),
        ),
        dir_preds=os.path.join(
            cfg.root_dir,
            "data/predictions2/%s_preds%s.hdf5" % (output_prefix, indexstr),
        ),
    )

    get_model_preds(CP)

    elapsed_time = time.time() - start_time
    logger.info("Time elapsed: {0:.1f} min".format(elapsed_time / 60.0))


@predict.command()
@click.argument("llt2", type=float, default=cfg.longlat_thresh2_)
@click.argument("rt", type=float, default=cfg.rad_thresh_)
@click.option("--index", type=int, default=None)
@click.option("--prefix", default="ran2")
@click.option("--start", default=-1)
@click.option("--stop", default=-1)
@click.option("--matches", is_flag=True, default=False)
@click.option("--model", default=None)
def make_prediction(llt2, rt, index, prefix, start, stop, matches, model):
    """ Make predictions.

    Make predictions from a dataset,
    optionally using the precalculated CNN predictions.

    """
    logger = logging.getLogger(__name__)
    logger.info("making predictions.")
    start_time = time.time()
    if index is None:
        indexstr = ""
    else:
        indexstr = "_{:05d}".format(index)

    # Crater Parameters
    CP = {}

    # Image width/height, assuming square images.
    CP["dim"] = 256

    # Data type - train, dev, test
    CP["datatype"] = prefix

    # Number of images to extract craters from
    CP["n_imgs"] = -1  # all of them

    # Hyperparameters
    CP["llt2"] = llt2  # D_{L,L} from Silburt et. al (2019)
    CP["rt"] = rt  # D_{R} from Silburt et. al (2019)

    # Location of model to generate predictions (if they don't exist yet)
    CP["dir_model"] = model

    # Location of where hdf5 data images are stored
    CP["dir_data"] = os.path.join(
        cfg.root_dir,
        "data/processed/%s_images%s.hdf5" % (CP["datatype"], indexstr),
    )
    # Location of where model predictions are/will be stored
    CP["dir_preds"] = os.path.join(
        cfg.root_dir,
        "data/predictions2/%s_preds%s.hdf5" % (CP["datatype"], indexstr),
    )
    # Location of where final unique crater distribution will be stored
    CP["dir_result"] = os.path.join(
        cfg.root_dir,
        "data/predictions2/%s_craterdist%s.npy" % (CP["datatype"], indexstr),
    )
    # Location of hdf file containing craters found
    CP["dir_craters"] = os.path.join(
        cfg.root_dir,
        "data/predictions2/%s_craterdist%s.hdf5" % (CP["datatype"], indexstr),
    )
    # Location of hdf file containing craters found
    CP["dir_input_craters"] = os.path.join(
        cfg.root_dir,
        "data/processed/%s_craters%s.hdf5" % (CP["datatype"], indexstr),
    )

    craters_unique = np.empty([0, 3])

    craters_unique = extract_unique_craters(
        CP, craters_unique, index=index, start=start, stop=stop, withmatches=matches
    )

    elapsed_time = time.time() - start_time
    logger.info("Time elapsed: {0:.1f} min".format(elapsed_time / 60.0))


def make_global_statistics(CP):
    my_craters_h5 = pd.HDFStore(CP['dir_craters'])
    my_craters = my_craters_h5['all']
    robbins_h5 = pd.HDFStore(CP['dir_input_craters'])
    
    robbins_filtered = pd.concat([
            robbins_h5[k][(((robbins_h5[k]['x (pix)']-128).abs() < 128) & 
                           ((robbins_h5[k]['y (pix)']-128).abs() < 128) &
                           (robbins_h5[k]["Diameter (pix)"] < 2*cfg.maxrad_) &
                           (robbins_h5[k]["Diameter (pix)"] > 2*cfg.minrad_))]
            for k in robbins_h5.keys()])
    cols = ["Lat","Long","Diameter (km)"]
    robbins_filtered = robbins_filtered[cols]
    
    robbins_unique = metric.rep_filter_unique_craters(robbins_filtered,
                                                      *cols)[0]
    
    matched_craters = metric.kn_match_craters(my_craters, robbins_unique,
                                              *cols)[0]
    
    N_match = len(matched_craters)
    N_robbins = len(robbins_unique)
    N_detect = len(my_craters)
    print(N_match, N_robbins, N_detect)
    precision = N_match / N_detect
    recall = N_match / N_robbins
    fscore = 2 * precision * recall / (precision + recall)
    
    robbins_h5.close()
    my_craters_h5.close()
    
    return precision, recall, fscore


@predict.command()
@click.argument("llt2", type=float, default=cfg.longlat_thresh2_)
@click.argument("rt", type=float, default=cfg.rad_thresh_)
@click.option("--index", type=int, default=None)
@click.option("--prefix", default="ran2")
@click.option("--start", default=-1)
@click.option("--stop", default=-1)
@click.option("--matches", is_flag=True, default=False)
@click.option("--model", default=None)
def global_statistics(llt2, rt, index, prefix, start, stop, matches, model):
    logger = logging.getLogger(__name__)
    logger.info("making global statistics.")
    start_time = time.time()
    if index is None:
        indexstr = ""
    else:
        indexstr = "_{:05d}".format(index)
    
        # Crater Parameters
    CP = {}

    # Image width/height, assuming square images.
    CP["dim"] = 256

    # Data type - train, dev, test
    CP["datatype"] = prefix

    # Number of images to extract craters from
    CP["n_imgs"] = -1  # all of them

    # Hyperparameters
    CP["llt2"] = llt2  # D_{L,L} from Silburt et. al (2019)
    CP["rt"] = rt  # D_{R} from Silburt et. al (2019)

    # Location of model to generate predictions (if they don't exist yet)
    CP["dir_model"] = model

    # Location of where hdf5 data images are stored
    CP["dir_data"] = os.path.join(
        cfg.root_dir,
        "data/processed/%s_images%s.hdf5" % (CP["datatype"], indexstr),
    )
    # Location of where model predictions are/will be stored
    CP["dir_preds"] = os.path.join(
        cfg.root_dir,
        "data/predictions2/%s_preds%s.hdf5" % (CP["datatype"], indexstr),
    )
    # Location of where final unique crater distribution will be stored
    CP["dir_result"] = os.path.join(
        cfg.root_dir,
        "data/predictions2/%s_craterdist%s.npy" % (CP["datatype"], indexstr),
    )
    # Location of hdf file containing craters found
    CP["dir_craters"] = os.path.join(
        cfg.root_dir,
        "data/predictions2/%s_craterdist%s.hdf5" % (CP["datatype"], indexstr),
    )
    # Location of hdf file containing craters found
    CP["dir_input_craters"] = os.path.join(
        cfg.root_dir,
        "data/processed/%s_craters%s.hdf5" % (CP["datatype"], indexstr),
    )
    
    precision, recall, fscore = make_global_statistics(CP)
    
    logger.info('Precision: {:1.3f}'.format(precision))
    logger.info('Recall: {:1.3f}'.format(recall))
    logger.info('F Score: {:1.3f}'.format(fscore))
    
    elapsed_time = time.time() - start_time
    logger.info("Time elapsed: {0:.1f} min".format(elapsed_time / 60.0))

if __name__ == "__main__":
    predict()
