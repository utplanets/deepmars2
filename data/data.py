import numpy as np
import pandas as pd
from pyproj.transformer import Transformer
import tifffile
import h5py
import cv2
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import deepmars2.config as cfg

_R_Mars = 3389.5 # radius of Mars in km

def get_DEM(filename):
    """Reads the DEM from a large tiff file.

    Paramters
    ---------
    filename : str
        Path of the DEM file.

    Returns
    -------
    DEM : numpy.ndarray
        The image as a numpy array.
    """

    DEM = tifffile.imread(filename)
    
    # Remove missing data on left edge
    DEM[:,-1] = (DEM[:,-2] + DEM[:,0])/2
    
    return DEM


def get_IR(filename):
    """Reads the IR from a large tiff file.

    Paramters
    ---------
    filename : str
        Path of the IR file.

    Returns
    -------
    IR : numpy.ndarray
        The image as a numpy array.
    """
    
    return tifffile.imread(filename)


def get_craters(filename):
    """Reads the craters from a large tsv file.

    Paramters
    ---------
    filename : str
        Path of the craters file.

    Returns
    -------
    craters : pandas.DataFrame
        The list of craters.
    """
    
    craters = pd.read_csv(filename, sep='\t', engine='python')
    keep_columns = [
        'LATITUDE_CIRCLE_IMAGE',
        'LONGITUDE_CIRCLE_IMAGE',
        'DIAM_CIRCLE_IMAGE',
    ]
    craters = craters[keep_columns]
    craters.columns = ['Lat', 'Long', 'Diameter (km)']

    return craters


def fill_ortho_grid(lat_0, lon_0, box_size, img, dim=256):
    """Creates an orthographic projection from a plate caree projection.

    Paramters
    ---------
    lat_0 : float
        Central latitude of the image.
    lon_0 : float
        Central longitude of the image.
    box_size : float
        An abstract quantity measuring the size of the region being projected.
        It is proportional to the absolute size of the box in km but scaled so
        that at the equator, a box of size 1 denotes a box 1 degree across.
    img : numpy.ndarray
        The original image in plate carree coordinates to project from.
    dim : int, optional
        The width/height of the output image.  Only square outputs are
        supported.
    
    Returns
    -------
    ortho : numpy.ndarray
        The orthographic projection.
    """
    
    deg_per_pix = box_size / dim
    orthographic_coords = (np.indices((dim, dim)) - dim / 2) * deg_per_pix
    
    pipeline_str = (
        'proj=pipeline '
        'step proj=unitconvert xy_in=deg xy_out=rad '
        'step proj=eqc '
        'step proj=ortho inv lat_0={} lon_0={} '
        'step proj=unitconvert xy_in=rad xy_out=deg'
    ).format(lat_0, lon_0)
    
    transformer = Transformer.from_pipeline(pipeline_str)
    
    platecarree_coords = np.asarray(
        transformer.transform(orthographic_coords[0], orthographic_coords[1])
    )
    
    pixel_coords = np.asarray(
        [
            (90 - platecarree_coords[1, :, :]) * (img.shape[0] / 180),
            (platecarree_coords[0, :, :] - 180) * (img.shape[1] / 360),
        ]
    )
    
    pixel_coords = pixel_coords.astype(int)
    ortho = img[pixel_coords[0], pixel_coords[1]]
    
    return ortho


def make_mask(craters, dim, ring_size):
    """Creates a target mask given a list of craters.

    Paramters
    ---------
    craters : pandas.DataFrame
        A dataframe containing the crater positions in pixel space.  The
        columns must include 'x pix', 'y (pix)', and 'Diameter (pix)'.
    dim : int
        The width/height of the output image.  Only square outputs are
        supported.
    ring_size : int
        The thickness of the rings to be drawn.
    
    Returns
    -------
    ortho : numpy.ndarray
        The orthographic projection.
    """
    
    mask = np.zeros(shape=(dim, dim))
    
    if craters.empty:
        return mask, craters
    
    
    for irow, row in craters[['x (pix)', 'y (pix)', 'Diameter (pix)']].iterrows():
        cv2.circle(
            mask,
            (row['x (pix)'], row['y (pix)']),
            int(round(row['Diameter (pix)'] / 2)),
            255,
            ring_size,
        )

    return mask, craters


def normalize(array):
    """Normalize an array to have values between 0 and 1.
    
    Parameters
    ----------
    array : numpy.ndarray
        The array to be normalized
        
    Returns
    -------
    normalized : numpy.ndarray
        The normalized array.
    """
    
    
    shape = array.shape
    array = array.astype(np.float64)
    flattened = array.reshape(-1, 1)
    normalized = MinMaxScaler(feature_range=(0, 1)).fit_transform(flattened)
    normalized = normalized.reshape(shape)
    
    return normalized


def get_craters_in_img(craters, lat_0, lon_0, box_size, dim):
    """Return a list of the craters in an image.
    
    Parameters
    ----------
    craters : pandas.DataFrame
        The list of craters.  Must have columns including 'Lat', 'Long', and
        'Diameter (km)'.
    lat_0 : float
        Central latitude of the image.
    lon_0 : float
        Central longitude of the image.
    box_size : float
        An abstract quantity measuring the size of the region being projected.
        It is proportional to the absolute size of the box in km but scaled so
        that at the equator, a box of size 1 denotes a box 1 degree across.
    dim : int, optional
        The width/height of the output image.  Only square images are
        supported.
        
    Returns
    -------
    craters_in_img : pandas.DataFrame
        A list of craters in the image with columns 'Lat', 'Long',
        'Diameter (km)', 'x (pix)', 'y (pix)', and 'Diameter (pix)'.
    """
    
    # Rough overestimate
    craters_in_img_approx = craters[
        (np.abs(craters['Lat'] - lat_0) < box_size)
        & (np.abs(craters['Long'] - lon_0) < box_size / np.cos(np.deg2rad(lat_0)))
    ].copy()
    
    # Add new columns
    craters_in_img_approx['x (pix)'] = None
    craters_in_img_approx['y (pix)'] = None
    craters_in_img_approx['Diameter (pix)'] = None
    
    # Do nothing for empty crater list
    if len(craters_in_img_approx) == 0:
        return craters_in_img_approx
    
    # Convert from Lat/Long/km to pixels
    x, y, d = lld_to_xyd(craters_in_img_approx['Lat'].values.copy(),
                         craters_in_img_approx['Long'].values.copy(),
                         craters_in_img_approx['Diameter (km)'].values.copy(),
                         lat_0, lon_0, box_size)
    
    craters_in_img_approx['x (pix)'] = x
    craters_in_img_approx['y (pix)'] = y
    craters_in_img_approx['Diameter (pix)'] = d
    
    # Ensure craters are within image
    craters_in_img = craters_in_img_approx[
            (x >= 0) & (x < dim) &
            (y >= 0) & (y < dim) &
            (d >= 2 * cfg.minrad_) & (d <= 2 * cfg.maxrad_)
    ].copy()
    
    return craters_in_img


def xyd_to_lld(x, y, d, lat_0, lon_0, box_size, dim=256):  
    
    deg_per_pix = box_size / dim
    km_per_deg = np.pi * _R_Mars / 180
    km_per_pix = deg_per_pix * km_per_deg
    
    d *= km_per_pix
    
    x -= dim//2 # ensure that (0, 0) is in the center of the image
    y -= dim//2
    x *= deg_per_pix
    y *= deg_per_pix
    
    pipeline_str = (
        'proj=pipeline '
        'step proj=unitconvert xy_in=deg xy_out=rad '
        'step proj=eqc '
        'step proj=ortho inv lat_0={} lon_0={} '
        'step proj=unitconvert xy_in=rad xy_out=deg'
        ).format(lat_0, lon_0)
    transformer = Transformer.from_pipeline(pipeline_str)
    
    lon, lat = transformer.transform(y, x)
    
    return lat, lon, d


def lld_to_xyd(lat, lon, d, lat_0, lon_0, box_size, dim=256, return_ints=True):
    
    deg_per_pix = box_size / dim
    km_per_deg = np.pi * _R_Mars / 180
    km_per_pix = deg_per_pix * km_per_deg
    
    d /= km_per_pix
    
    pipeline_str = (
        'proj=pipeline '
        'step proj=unitconvert xy_in=deg xy_out=rad '
        'step proj=ortho lat_0={} lon_0={} '
        'step proj=eqc inv '
        'step proj=unitconvert xy_in=rad xy_out=deg'
        ).format(lat_0, lon_0)
    transformer = Transformer.from_pipeline(pipeline_str)
    
    y, x = transformer.transform(lon, lat)
    
    x /= deg_per_pix
    y /= deg_per_pix
    
    x += dim//2 # ensure that (0, 0) is in the center of the image
    y += dim//2
    
    if return_ints:
        return (np.round(x).astype(int),
                np.round(y).astype(int),
                np.round(d).astype(int))
    else:
        return x, y, d


def gen_dataset(
    DEM,
    IR,
    craters,
    series_prefix,
    start_index,
    amount=1000,
    dim=256,
    min_box_size=2,
    max_box_size=30,
    ring_size=1
):
    
    # Create HDF5 files
    imgs_filename = '{}/data/processed/{}_images_{:05d}.hdf5'.format(
            cfg.root_dir, series_prefix, start_index)
    imgs_h5 = h5py.File(imgs_filename, 'w')
    imgs_h5_DEM = imgs_h5.create_dataset('input_DEM',
                                         (amount, dim, dim),
                                         dtype='float32')
    imgs_h5_DEM.attrs['definition'] = 'Input DEM dataset.'
    imgs_h5_IR = imgs_h5.create_dataset('input_IR',
                                        (amount, dim, dim),
                                        dtype='float32')
    imgs_h5_IR.attrs['definition'] = 'Input IR dataset.'
    imgs_h5_targets = imgs_h5.create_dataset('target_masks',
                                             (amount, dim, dim),
                                             dtype='float32')
    imgs_h5_targets.attrs['definition'] = 'Target mask dataset.'
    imgs_h5_cll = imgs_h5.create_dataset('central_lat_lon',
                                         (amount, 2),
                                         dtype='float32')
    imgs_h5_cll.attrs['definition'] = 'Central latitude and longitude.'
    imgs_h5_box_size = imgs_h5.create_dataset('box_size',
                                              (amount, 1),
                                              dtype='float32')
    imgs_h5_box_size.attrs['definition'] = 'Box size'
    
    craters_filename = '{}/data/processed/{}_craters_{:05d}.hdf5'.format(
            cfg.root_dir, series_prefix, start_index)
    craters_h5 = pd.HDFStore(craters_filename)

    for i in tqdm(range(amount)):
        lat = np.random.uniform(-85, 85)
        lon = np.random.uniform(-180, 180)
        box_size = np.exp(np.random.uniform(np.log(min_box_size), np.log(max_box_size)))

        craters_in_img = get_craters_in_img(
            craters, lat, lon, box_size, dim
        )

        ortho_mask, craters_xy = make_mask(craters_in_img, dim, ring_size)
        ortho_DEM = fill_ortho_grid(lat, lon, box_size, DEM)
        ortho_IR = fill_ortho_grid(lat, lon, box_size, IR)

        #ortho_mask = normalize(ortho_mask)
        ortho_DEM = normalize(ortho_DEM)
        ortho_IR = normalize(ortho_IR)

        imgs_h5_DEM[i, ...] = ortho_DEM
        imgs_h5_IR[i, ...] = ortho_IR
        imgs_h5_targets[i, ...] = ortho_mask
        imgs_h5_cll[i, 0] = lat
        imgs_h5_cll[i, 1] = lon
        imgs_h5_box_size[i, 0] = box_size
        if craters_xy is not None:
            craters_h5['img_{:05d}'.format(start_index + i)] = craters_xy

        imgs_h5.flush()
        craters_h5.flush()

    imgs_h5.close()
    craters_h5.close()


def main():

    print('Loading DEM')
    DEM = get_DEM(cfg.DEM_filename)
    print('Loading IR')
    IR = get_IR(cfg.IR_filename)
    print('Loading craters')
    craters = get_craters(cfg.crater_filename)

    print('Generating dataset', flush=True)

    for i in range(1):
        start_index = i * 1000
        print('\n{:05d}'.format(start_index), flush=True)
        gen_dataset(DEM, IR, craters, 'ran3', start_index)


if __name__=='__main__':
    main()