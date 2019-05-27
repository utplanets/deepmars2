import numpy as np
import pandas as pd
from pyproj.transformer import Transformer
import tifffile
import h5py
import cv2
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from config import input_root, output_root


def get_DEM(filename):
    """Reads the DEM from a large tiff file.

    Paramters
    ----------
    filename : str, optional
        path of the DEM file, default to environment variable DM_MarsDEM

    Returns
    --------
    dem : numpy.ndarray
        The image as a numpy array.
    """

    return tifffile.imread(filename)


def get_IR(filename):

    return tifffile.imread(filename)


def get_craters(filename):

    craters = pd.read_csv(filename, sep="\t", engine="python")
    keep_columns = [
        "LATITUDE_CIRCLE_IMAGE",
        "LONGITUDE_CIRCLE_IMAGE",
        "DIAM_CIRCLE_IMAGE",
    ]
    craters = craters[keep_columns]
    craters.columns = ["Lat", "Long", "Diameter (km)"]

    return craters


def fill_ortho_grid(lat, lon, box_size, img, dim=256):

    deg_per_pix = box_size / dim
    orthographic_coords = (np.indices((dim, dim)) - dim / 2) * deg_per_pix
    pipeline_str = (
        "proj=pipeline "
        "step proj=unitconvert xy_in=deg xy_out=rad "
        "step proj=eqc "
        "step proj=ortho inv lat_0={} lon_0={} "
        "step proj=unitconvert xy_in=rad xy_out=deg"
    ).format(lat, lon)
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

    return img[pixel_coords[0], pixel_coords[1]]


def make_mask(lat, lon, craters, box_size, dim, ring_size):

    mask = np.zeros(shape=(dim, dim))

    if craters.empty:
        return mask, None

    pipeline_str = (
        "proj=pipeline "
        "step proj=unitconvert xy_in=deg xy_out=rad "
        "step proj=ortho lat_0={} lon_0={} "
        "step proj=eqc inv "
        "step proj=unitconvert xy_in=rad xy_out=deg"
    ).format(lat, lon)
    transformer = Transformer.from_pipeline(pipeline_str)

    pix_per_deg = dim / box_size
    pix_per_km = pix_per_deg * 360 / (2 * np.pi * 3389.5)

    craters["Diameter (pix)"] = craters["Diameter (km)"] * pix_per_km
    craters["y (deg)"], craters["x (deg)"] = transformer.transform(
        craters["Long"].values, craters["Lat"].values
    )
    craters["x (pix)"] = craters["x (deg)"] * pix_per_deg
    craters["y (pix)"] = craters["y (deg)"] * pix_per_deg
    craters["Diameter (pix)"] = craters["Diameter (pix)"].astype(int)
    craters[craters == np.inf] = 1000000
    craters["x (pix)"] = craters["x (pix)"].astype(int) + dim // 2
    craters["y (pix)"] = craters["y (pix)"].astype(int) + dim // 2

    for irow, row in craters[["x (pix)", "y (pix)", "Diameter (pix)"]].iterrows():
        cv2.circle(
            mask,
            (row["x (pix)"], row["y (pix)"]),
            row["Diameter (pix)"] // 2,
            255,
            ring_size,
        )

    return mask, craters


def normalize(array):

    shape = array.shape
    array = array.astype(np.float64)
    flattened = array.reshape(-1, 1)
    normalized = MinMaxScaler(feature_range=(0, 1)).fit_transform(flattened)

    return normalized.reshape(shape)


def get_craters_in_img(craters, lat, lon, box_size, dim, min_diameter_pix):

    pix_per_km = (dim / box_size) * 360 / (2 * np.pi * 3389.5)
    craters_in_img = craters[
        (np.abs(craters["Lat"] - lat) < box_size/2)
        & (np.abs(craters["Long"] - lon) < box_size/2 / np.cos(np.deg2rad(lat)))
        & (craters["Diameter (km)"] * pix_per_km > min_diameter_pix)
        & (craters["Diameter (km)"] * pix_per_km < box_size)
    ]

    return pd.DataFrame(craters_in_img)


def gen_dataset(
    DEM,
    IR,
    craters,
    series_prefix,
    start_index=0,
    amount=100,
    dim=256,
    min_box=2,
    max_box=30,
    min_diameter_pix=4,
    ring_size=1,
):

    imgs_h5 = h5py.File(
        ("{}/data/processed/{}_images_{:05d}.hdf5").format(
            output_root, series_prefix, start_index
        ),
        "w",
    )
    imgs_h5_DEM = imgs_h5.create_dataset(
        "input_DEM", (amount, dim, dim), dtype="float32"
    )
    imgs_h5_DEM.attrs["definition"] = "Input DEM dataset."
    imgs_h5_IR = imgs_h5.create_dataset("input_IR", (amount, dim, dim), dtype="float32")
    imgs_h5_IR.attrs["definition"] = "Input IR dataset."
    imgs_h5_targets = imgs_h5.create_dataset(
        "target_masks", (amount, dim, dim), dtype="float32"
    )
    imgs_h5_targets.attrs["definition"] = "Target mask dataset."
    imgs_h5_cll = imgs_h5.create_dataset(
        "central_lat_lon", (amount, 2), dtype="float32"
    )
    imgs_h5_cll.attrs["definition"] = "Central latitude and longitude."
    imgs_h5_box_size = imgs_h5.create_dataset("box_size", (amount, 1), dtype="float32")
    imgs_h5_box_size.attrs["definition"] = "Box size"
    craters_h5 = pd.HDFStore(
        ("{}/data/processed/{}_craters_{:05d}.hdf5").format(
            output_root, series_prefix, start_index
        )
    )

    for i in tqdm(range(amount)):
        lat = np.random.uniform(-85, 85)
        lon = np.random.uniform(-180, 180)
        box_size = np.exp(np.random.uniform(np.log(min_box), np.log(max_box)))

        craters_in_img = get_craters_in_img(
            craters, lat, lon, box_size, dim, min_diameter_pix
        )

        ortho_mask, craters_xy = make_mask(
            lat, lon, craters_in_img, box_size, dim, ring_size
        )
        ortho_DEM = fill_ortho_grid(lat, lon, box_size, DEM)
        ortho_IR = fill_ortho_grid(lat, lon, box_size, IR)

        ortho_mask = normalize(ortho_mask)
        ortho_DEM = normalize(ortho_DEM)
        ortho_IR = normalize(ortho_IR)

        imgs_h5_DEM[i, ...] = ortho_DEM
        imgs_h5_IR[i, ...] = ortho_IR
        imgs_h5_targets[i, ...] = ortho_mask
        imgs_h5_cll[i, 0] = lat
        imgs_h5_cll[i, 1] = lon
        imgs_h5_box_size[i, 0] = box_size
        if craters_xy is not None:
            craters_h5["img_{:05d}".format(start_index + i)] = craters_xy

        imgs_h5.flush()
        craters_h5.flush()

    imgs_h5.close()
    craters_h5.close()


_dem_filename = input_root + "/data/raw/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
_ir_filename = input_root + "/data/raw/Mars_THEMIS_scaled.tif"
_crater_filename = input_root + "/data/raw/RobbinsCraters_20121016.tsv"


def main(
    dem_filename=_dem_filename,
    ir_filename=_ir_filename,
    crater_filename=_crater_filename,
):

    print("Loading DEM")
    DEM = get_DEM(dem_filename)
    print("Loading IR")
    IR = get_IR(ir_filename)
    print("Loading craters")
    craters = get_craters(crater_filename)

    print("Generating dataset", flush=True)

    for i in range(50):
        start_index = i * 1000
        print("\n{:05d}".format(start_index), flush=True)
        gen_dataset(DEM, IR, craters, "ran", amount=1000, start_index=start_index)


if __name__=="__main__":
    main()
