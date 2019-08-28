import numpy as np
import pandas as pd
from pyproj.transformer import Transformer
import tifffile
import h5py
import cv2
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
import deepmars2.config as cfg


def get_DEM(filename, is_Mars):
    """Reads the DEM from a large tiff file.

    Paramters
    ---------
    filename : str
        Path of the DEM file.
    is_Mars : bool, optional
        Whether the DEM is of Mars
    
    Returns
    -------
    DEM : numpy.ndarray
        The image as a numpy array.
    """

    DEM = tifffile.imread(filename)
    
    if is_Mars:
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
    
    IR = tifffile.imread(filename)
    
    return IR


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


def get_ortho_grid(lat_0, lon_0, box_size, dim=256):
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
    return platecarree_coords


def draw_ortho_grid(platecarree_coords, img1, img2, dim=256):
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
    
    input_imgs = []
    for img in [img1, img2]:
        pixel_coords = np.asarray(
            [
                (90 - platecarree_coords[1, :, :]) * (img.shape[0] / 180),
                (platecarree_coords[0, :, :] - 180) * (img.shape[1] / 360),
            ]
        )
    
        pixel_coords = pixel_coords.astype(int)
        input_imgs.append(
            normalize(img[pixel_coords[0], pixel_coords[1]])
            )

    ortho = np.dstack(input_imgs).reshape((1,dim,dim,2))
    
    return ortho
    #return pixel_coords

def make_mask(craters, ring_size, dim=256):
    """Creates a target mask given a list of craters.

    Paramters
    ---------
    craters : pandas.DataFrame
        A dataframe containing the crater positions in pixel space.  The
        columns must include 'x pix', 'y (pix)', and 'Diameter (pix)'.
    ring_size : int
        The thickness of the rings to be drawn.
    dim : int, optional
        The width/height of the output image.  Only square outputs are
        supported.
    
    Returns
    -------
    mask : numpy.ndarray
        The target mask.
    """
    
    mask = np.zeros(shape=(dim, dim))
    
    if craters.empty:
        return mask
    
    
    for irow, row in craters[['x (pix)', 'y (pix)', 'Diameter (pix)']].iterrows():
        cv2.circle(
            mask,
            (row['x (pix)'], row['y (pix)']),
            int(round(row['Diameter (pix)'] / 2)),
            255,
            ring_size,
        )

    return mask


def normalize(array):
    """Normalize an array to have values between 0 and 1.
    
    Parameters
    ----------
    array : numpy.ndarray
        The array to be normalized.
        
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


def get_craters_in_img(craters, lat_0, lon_0, box_size, dim=256):
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
    """Convert crater coordinates from pixels in an orthographic projection
    to latitude and longitude in degrees and diameter in kilometers.
    
    Parameters
    ----------
    x : int
        The x-coordinate in pixels.
    y : int
        The y-coordinate in pixels.
    d : int
        The diameter in pixels.
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
    lat : float
        The latitude in degrees.
    lon : float
        The longitude in degrees.
    d : float
        The diameter in kilometers.
    """
    
    deg_per_pix = box_size / dim
    km_per_deg = np.pi * cfg.R_planet / 180
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
    """Convert crater coordinates from latitude and longitude in degrees and
    diameter in kilometers to pixels in an orthographic projection.
    
    Parameters
    ----------
    lat : float
        The latitude in degrees.
    lon : float
        The longitude in degrees.
    d : float
        The diameter in kilometers.
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
    x : int
        The x-coordinate in pixels.
    y : int
        The y-coordinate in pixels.
    d : int
        The diameter in pixels.
    """
    
    deg_per_pix = box_size / dim
    km_per_deg = np.pi * cfg.R_planet / 180
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


def get_approx_width(box_size, lat):
    """Get the approximate width of a box at a given latitude and box size.
    
    Parameters
    ----------
    box_size : float
        An abstract quantity measuring the size of the region being projected.
    It is proportional to the absolute size of the box in km but scaled so
    that at the equator, a box of size 1 denotes a box 1 degree across.
    lat : float
        The latitude of the box in degrees.
    
    Returns
    -------
    min_width : float
        The approximate minimum width of the box.
    """
    
    min_width = box_size / np.cos(np.deg2rad(lat))
    
    return min_width


def systematic_pass(box_sizes, min_lat=-90, max_lat=90, min_long=-180, max_long=180):
    coords = []
    
    for box_size in box_sizes:
        box_size = box_size / 2 # for overlap
        n_lats = int(np.ceil((max_lat - min_lat) / box_size))
        lats = np.linspace(min_lat, max_lat, n_lats + 1)
        lats = lats[:-1] + np.diff(lats) / 2
        
        for lat in lats:
            width = get_approx_width(box_size, lat)
            n_lons = int(np.ceil((max_long - min_long) / width))
            lons = np.linspace(min_long, max_long, n_lons + 1)
            lons = lons[:-1] + np.diff(lons) / 2
            
            for lon in lons:
                coords.append([lat, lon, box_size * 2]) # rescale box_size
    
    print('{} images'.format(len(coords)))
    
    return np.array(coords)
            

def make_images(craters, lat, lon, box_size, dim, DEM, IR, ring_size):
    craters_in_img = get_craters_in_img(craters, lat, lon, box_size, dim=dim)
    
    ortho_mask = make_mask(craters_in_img, ring_size, dim=dim)
    ortho_DEM = fill_ortho_grid(lat, lon, box_size, DEM)
    ortho_IR = fill_ortho_grid(lat, lon, box_size, IR)

    ortho_mask = normalize(ortho_mask)
    ortho_DEM = normalize(ortho_DEM)
    ortho_IR = normalize(ortho_IR)
    
    return ortho_DEM, ortho_IR, ortho_mask, craters_in_img
        


def gen_dataset(
    DEM,
    IR,
    craters,
    series_prefix,
    start_index,
    mode,
    sys_pass=None,
    amount=1000,
    dim=256,
    min_box_size=2,
    max_box_size=30,
    ring_size=1,
    in_notebook=False,
    min_lat=-90,
    max_lat=90,
    min_long=-180,
    max_long=180
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

    if in_notebook:
        tqdm_type = tqdm_notebook
    else:
        tqdm_type = tqdm
    
    for i in tqdm_type(range(amount)):
        if mode=='random':
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_long, max_long)
            box_size = np.exp(np.random.uniform(np.log(min_box_size), np.log(max_box_size)))

        elif mode=='systematic':
            if sys_pass is None:
                raise ValueError('If mode is systematic then sys_pass must be provided')
            
            elif start_index + i >= len(sys_pass):
                imgs_h5.flush()
                craters_h5.flush()
                imgs_h5.close()
                craters_h5.close()
                print('\nNo more images', flush=True)
                return
            else:
                lat, lon, box_size = sys_pass[start_index + i]
        
        else:
            raise ValueError('Mode must be either random or systematic')
            
        ortho_DEM, ortho_IR, ortho_mask, craters_xy = make_images(craters, lat,
                                                                  lon,
                                                                  box_size,
                                                                  dim, DEM, IR,
                                                                  ring_size)

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
    #DEM = get_DEM(cfg.DEM_filename)
    #DEM = tifffile.imread('/disks/work/james/deepmars2/data/raw/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif')
    DEM = tifffile.imread('../data/raw/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif')
    padding = (DEM.shape[1] // 2 - DEM.shape[0]) // 2
    new_DEM = np.zeros((DEM.shape[1] // 2, DEM.shape[1]), dtype='int16')
    new_DEM[padding:padding + DEM.shape[0],:] = DEM
    del DEM
    DEM = new_DEM
    print('Loading IR')
    #IR = get_IR(cfg.IR_filename)
    #IR = tifffile.imread('/disks/work/james/deepmars2/data/raw/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif')
    IR = tifffile.imread('../data/raw/Lunar_LRO_LOLAKaguya_Shade_60N60S_512ppd.tif')
    padding = (IR.shape[1] // 2 - IR.shape[0]) // 2
    new_IR = np.zeros((IR.shape[1] // 2, IR.shape[1]), dtype='int16')
    new_IR[padding:padding + IR.shape[0],:] = IR
    del IR
    IR = new_IR
    print('Loading craters')
    #craters = get_craters(cfg.crater_filename)
    # Load LROC Craters (5km - 20km)
    
    #cols = ['Long', 'Lat', 'Diameter (km)']
    #LROC = pd.read_csv('../data/raw/LROCCraters.csv')[cols].copy()
    
    # Load Head Craters (20 km +)
    
    #Head = pd.read_csv('../data/raw/HeadCraters.csv')
    #Head.rename(columns={'Lon':'Long', 'Lat':'Lat', 'Diam_km':'Diameter (km)'}, inplace=True)
    #Head = Head[cols].copy()
    
    #craters = pd.concat([LROC, Head])
     
    # Moon Robbins
    
    cols = ['Long', 'Lat', 'Diameter (km)']
    robbins_cols = ['LON_CIRC_IMG', 'LAT_CIRC_IMG', 'DIAM_CIRC_IMG']
    Robbins = pd.read_csv('../data/raw/lunar_crater_database_robbins_2018.csv')[robbins_cols]
    Robbins.rename(columns=dict((robbins_cols[i],cols[i]) for i in range(3)), inplace=True)
    Robbins.loc[Robbins['Long'] > 180, 'Long'] -= 360
    
    craters = Robbins
    
    print('Generating dataset', flush=True)
    
    for i in range(50):
        start_index = i * 1000
        print('\n{:05d}'.format(start_index), flush=True)
        gen_dataset(DEM, IR, craters, 'ran_moon_hd_robbins', start_index, 'random', min_box_size=1.7, min_lat=-60, max_lat=60)


if __name__=='__main__':
    main()
