import time
import deepmars2.post_processing_net.model as ppn
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
from subprocess import Popen
import matplotlib.pyplot as plt
from sklearn.neighbors import RadiusNeighborsClassifier
from joblib import Parallel, delayed
import tifffile
from tqdm import tqdm
from keras.models import load_model
from multiprocessing import Process, Queue
import deepmars2.data.data as data
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import deepmars2.config as cfg
import cratertools.metric as metric
import sys


cols = ['Long', 'Lat', 'Diameter (km)']


# Customizable for individual dataset

# Moon Robbins

min_box_size = 1.7
max_box_size = 30
n_box_sizes = 20
min_lat = -60
max_lat = 60
min_long = -180
max_long = 180
min_diam = 0
max_diam = np.inf
t_rad = 0.25
t_ll2 = 0.5

def load_DEM_IR():
    print('Loading DEM')
    DEM = tifffile.imread('./data/raw/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif')
    padding = (DEM.shape[1] // 2 - DEM.shape[0]) // 2
    new_DEM = np.zeros((DEM.shape[1] // 2, DEM.shape[1]), dtype='int16')
    new_DEM[padding:padding + DEM.shape[0],:] = DEM
    
    print('Loading IR')
    IR = tifffile.imread('./data/raw/Lunar_LRO_LOLAKaguya_Shade_60N60S_512ppd.tif')
    padding = (IR.shape[1] // 2 - IR.shape[0]) // 2
    new_IR = np.zeros((IR.shape[1] // 2, IR.shape[1]), dtype='int16')
    new_IR[padding:padding + IR.shape[0],:] = IR
    
    return new_DEM, new_IR

def load_craters():
    print('Loading Craters')
    robbins_cols = ['LON_CIRC_IMG', 'LAT_CIRC_IMG', 'DIAM_CIRC_IMG']
    Robbins = pd.read_csv('./data/raw/lunar_crater_database_robbins_2018.csv')[robbins_cols]
    Robbins.rename(columns=dict((robbins_cols[i], cols[i]) for i in range(3)), inplace=True)
    Robbins.loc[Robbins['Long'] > 180, 'Long'] -= 360
    
    return Robbins

# Moon P&H
#
#min_box_size = 4
#max_box_size = 30
#n_box_sizes = 20
#min_lat = -90
#max_lat = 90
#min_long = -180
#max_long = 180
#min_diam = 0
#max_diam = np.inf
#t_rad = 0.25
#t_ll2 = 0.5


#def load_DEM_IR():
#    print('Loading DEM')
#    DEM = tifffile.imread('./data/raw/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif')
#    print('Loading IR')
#    IR = tifffile.imread('./data/raw/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif')
#    return DEM, IR

#def load_craters():
#    LROC = pd.read_csv('./data/raw/LROCCraters.csv')[cols].copy()
#    Head = pd.read_csv('./data/raw/HeadCraters.csv')
#    Head.rename(columns={'Lon':'Long', 'Lat':'Lat', 'Diam_km':'Diameter (km)'}, inplace=True)
#    
#    return pd.concat([LROC, Head])
    

# helper functions

def get_crater_imgs(crater, DEM, IR, padding=1.2):
    long, lat, diam = crater[cols]
    deg_per_km = 180 / (np.pi * cfg.R_planet)
    box_size = diam * deg_per_km * padding
    img_DEM = data.fill_ortho_grid(lat, long, box_size, DEM)
    img_IR = data.fill_ortho_grid(lat, long, box_size, IR)
    img_DEM = data.normalize(img_DEM)
    img_IR = data.normalize(img_IR)
    return img_DEM, img_IR

def new_crater_imgs(mode, min_d, max_d, DEM, IR, craters):
    if mode=='positive':
        size_range = (craters['Diameter (km)'] > min_d) & (craters['Diameter (km)'] < max_d)
        crater = craters[size_range].sample().iloc[0].copy()
    elif mode=='negative':
        size_range = (craters['Diameter (km)'] > min_d) & (craters['Diameter (km)'] < max_d)
        crater = craters[size_range].sample().iloc[0].copy()
        # modify real crater
        diam = crater['Diameter (km)']
        deg_per_km = 180 / (np.pi * cfg.R_planet)
        DeltaDiam = np.random.uniform(-0.5, 0.5) * diam
        DeltaLat = np.random.uniform(-1, 1) * diam * deg_per_km
        DeltaLong = np.random.uniform(-1, 1) * diam * deg_per_km
        crater['Diameter (km)'] += DeltaDiam
        crater['Lat'] += DeltaLat
        crater['Long'] += DeltaLong
    else:
        raise ValueError('Mode must be one of positive or negative')
    
    img_DEM, img_IR = get_crater_imgs(crater, DEM, IR)
    
    return img_DEM, img_IR, crater

def rotate(img):
    return img.T[::-1,:]

def display_imgs(img_DEM, img_IR):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(rotate(img_DEM), 'Greys_r')
    axs[1].imshow(rotate(img_IR), 'Greys_r')
    axs[0].set_title('DEM')
    axs[1].set_title('IR')
    plt.show()

def generator(batch_size, min_d, max_d, DEM, IR, craters, dim=256):
    while True:
        targets = np.random.randint(2, size=(batch_size,1))
        input_imgs = np.zeros(shape=(batch_size, dim, dim, 2))
        for i in range(batch_size):
            if targets[i,0] == 1:
                mode = 'positive'
            else:
                mode = 'negative'
            img_DEM, img_IR, crater = new_crater_imgs(mode, min_d, max_d, DEM, IR, craters)
            input_imgs[i,:,:,0] = img_DEM
            input_imgs[i,:,:,1] = img_IR
        yield input_imgs, targets

def get_crater_prob(crater, model, DEM, IR):
    img_DEM, img_IR = get_crater_imgs(crater, DEM, IR)
    input_imgs = np.zeros((1,256, 256, 2))
    input_imgs[0,:,:,0] = img_DEM
    input_imgs[0,:,:,1] = img_IR
    preds = model.predict(input_imgs)
    return preds[0,0]

def add_crater_probs(craters_df, model, DEM, IR):
    probs = np.zeros(len(craters_df))
    for i in range(len(craters_df)):
        crater = craters_df.iloc[i]
        probs[i] = get_crater_prob(crater, model, DEM, IR)
    craters_df['prob'] = probs
    
def filter_df(df, min_lat=-90, max_lat=90, min_long=-180, max_long=180, min_diam=0, max_diam=np.inf):
    return df[(df['Lat'] > min_lat) & (df['Lat'] < max_lat) & 
              (df['Long'] > min_long) & (df['Long'] < max_long) &
              (df['Diameter (km)'] > min_diam) & (df['Diameter (km)'] < max_diam)].copy()

def is_match(crater1, crater2, thresh_rad=0.25, thresh_ll2=0.5):
    long1, lat1, diam1 = crater1[:]
    long2, lat2, diam2 = crater2[:]
    
    min_d = min(diam1, diam2)
    dR = np.abs(diam1 - diam2) / min_d
    
    k2d = 180 / (np.pi * cfg.R_planet)
    lat_m = (lat1 + lat2) / 2
    dL = (((long1 - long2) * np.cos(np.deg2rad(lat_m)) / (0.5 * k2d * min_d))**2 +
          ((lat1 - lat2) / (0.5 * k2d * min_d))**2)
    
    return dL < thresh_ll2 and dR < thresh_rad

def find(k, S, N, visited):
    L = N[k]
    S.extend(L)
    visited.add(k)
    for elem in L:
        if elem not in visited:
            find(elem, S, N, visited)

def S_to_crater(S, craters_np):
    craters = np.array(craters_np[S])
    return craters.mean(axis=0)

def filter_craters(N, craters_np):
    visited = set()
    new_craters = []
    for k in tqdm(N.keys()):
        if k not in visited:
            S = []
            find(k, S, N, visited)
            crater = S_to_crater(S, craters_np)
            new_craters.append(crater)
    return np.array(new_craters)

def merge_dicts(dicts):
    new_dict = dict()
    for d in tqdm(dicts):
        for k in d.keys():
            if k not in new_dict.keys():
                new_dict[k] = d[k]
            else:
                new_dict[k] = np.hstack([new_dict[k], d[k]])
    
    # Filter out duplicates:
    for k in tqdm(new_dict.keys()):
        new_dict[k] = np.unique(new_dict[k])
    return new_dict

def radius_nbrs(bin_min, rescaled, batch_size):
    bin_max = bin_min * (1 + t_rad)**2
    thresh = t_ll2 * bin_max

    bin_craters = rescaled[(rescaled['d'] >= bin_min) & (rescaled['d'] <= bin_max)].copy()
    
    bin_indices = np.array(bin_craters.index)
    
    bin_craters_ll = np.array(bin_craters[['lat', 'long']])
    if len(bin_craters) == 0:
        return dict()
    
    nbrs = RadiusNeighborsClassifier(radius=thresh, metric='haversine')
    nbrs.fit(bin_craters_ll, np.zeros(len(bin_craters_ll)))
    
    
    neighbours = []
    
    def f(batch, nbrs, bin_indices, start):
        indices = nbrs.radius_neighbors(batch, return_distance=False)
        return dict((bin_indices[k + start], bin_indices[indices[k]]) for k in range(len(indices)))
    
    n_batches = int(np.ceil(len(bin_craters_ll) / batch_size))
    
    neighbours = Parallel(n_jobs=16)(delayed(f)(bin_craters_ll[i*batch_size:(i+1)*batch_size], nbrs, bin_indices, i*batch_size)
                          for i in tqdm(range(n_batches)))

    return {k: v for d in neighbours for k, v in d.items()}

def my_proj(ortho_coords, lat_0, lon_0):
    deg_to_rad = np.pi / 180
    R = 1
    x = ortho_coords[0] * deg_to_rad
    y = ortho_coords[1] * deg_to_rad
    phi_0 = lat_0 * deg_to_rad
    lamb_0 = lon_0 * deg_to_rad
    rho = np.sqrt(x**2 + y**2)
    c = np.arcsin(rho / R)
    cos_phi_0 = np.cos(phi_0)
    sin_phi_0 = np.sin(phi_0)
    cos_c = np.cos(c)
    sin_c = np.sin(c)
    eps=1e-7
    phi = np.arcsin(cos_c * sin_phi_0 + y * sin_c * cos_phi_0 / np.maximum(rho, eps))
    lamb = lamb_0 + np.arctan(x * sin_c / np.maximum(rho * cos_c * cos_phi_0 - y * sin_c * sin_phi_0, eps))
    lat = phi / deg_to_rad
    lon = lamb / deg_to_rad
    return lon, lat

def my_fill_ortho_grid(lat_0, lon_0, box_size, img, dim=256):
    deg_per_pix = box_size / dim
    ortho_coords = (np.indices((dim, dim)) - dim / 2) * deg_per_pix
    lon, lat = my_proj(ortho_coords, lat_0, lon_0)
    if img is None:
        return lon, lat
    x = ((90 - lat) * (img.shape[0] / 180)).astype(int)
    y = ((lon - 180) * (img.shape[1] / 360)).astype(int)
    ortho = img[x, y]
    return ortho

def make_batch(craters, DEM_shape, IR_shape):
    padding = 1.2
    deg_per_km = 180 / (np.pi * cfg.R_planet)
    if type(craters) == pd.core.frame.DataFrame:
        craters_np = np.array(craters[cols])
    elif type(craters) == np.ndarray:
        craters_np = craters
    else:
        raise ValueError('')
    batch_size = len(craters_np)
    batch = np.zeros((batch_size, 256, 256, 4))
    IR_DEM_same_shape = IR_shape == DEM_shape
    for i in range(batch_size):
        long, lat, diam = craters_np[i]
        box_size = diam * deg_per_km * padding
        lon, lat = my_fill_ortho_grid(lat, long, box_size, None)
        x_DEM = ((90 - lat) * (DEM_shape[0] / 180)).astype(int)
        y_DEM = ((lon - 180) * (DEM_shape[1] / 360)).astype(int)
        if not IR_DEM_same_shape:
            x_IR = ((90 - lat) * (IR_shape[0] / 180)).astype(int)
            y_IR = ((lon - 180) * (IR_shape[1] / 360)).astype(int)
        else:
            x_IR = x_DEM
            y_IR = y_DEM
        batch[i] = np.dstack([x_DEM, y_DEM, x_IR, y_IR])
    return batch

def my_norm(a):
    M = a.max(axis=(1,2))
    m = a.min(axis=(1,2))
    delta = np.maximum(M - m, 1e-7)
    return np.divide(a - m[:,np.newaxis, np.newaxis], delta[:,np.newaxis, np.newaxis])

def memory_worker(mem_queue, cpu_queues, gpu_to_mem_queue, gpu_queue):
    #load datasets
    
    DEM, IR = load_DEM_IR()
    
    # send shapes
    for q in cpu_queues:
        q.put((DEM.shape, IR.shape))
    
    # send ok signal
    for q in cpu_queues:
        q.put('ok')
    
    n_cpus = len(cpu_queues)
    n_cpus_done = 0
    
    while n_cpus_done < n_cpus:
        #print('waiting for signal (gpu)', flush=True)
        signal = mem_queue.get()
        if signal == 'done':
            print('received done signal')
            n_cpus_done += 1
            continue
        
        #print('signal received (gpu)', flush=True)
        if signal == 'stop':
            return
        # otherwise signal is the index of the cpu_worker that has a job followed by the start_index
        cpu_index, start_index = signal
        filename = '/mnt/ramdisk/coords_{}.npy'.format(cpu_index)
        coords = np.load(filename)
        
        # tell cpu worker that it's safe to send more coords
        #print('sending ok signal (gpu)', flush=True)
        cpu_queues[cpu_index].put('ok')
        
        # convert coords into imgs from DEM and IR
        x_DEM = coords[:,:,:,0]
        y_DEM = coords[:,:,:,1]
        x_IR = coords[:,:,:,2]
        y_IR = coords[:,:,:,3]
        
        try:
            imgs_DEM = DEM[x_DEM, y_DEM]
            imgs_IR = IR[x_IR, y_IR]
        except IndexError:
            x_DEM = np.mod(x_DEM, DEM.shape[0])
            y_DEM = np.mod(y_DEM, DEM.shape[1])
            x_IR = np.mod(x_IR, IR.shape[0])
            y_IR = np.mod(y_IR, IR.shape[1])
            imgs_DEM = DEM[x_DEM, y_DEM]
            imgs_IR = IR[x_IR, y_IR]
        
        imgs_DEM = my_norm(imgs_DEM)
        imgs_IR = my_norm(imgs_IR)
        
        imgs = np.stack([imgs_DEM, imgs_IR], axis=-1)
        #print(imgs.mean())
        
        # wait for gpu signal then save
        #print('waiting for signal 2 (gpu)', flush=True)
        signal = gpu_to_mem_queue.get()
        #print('signal 2 received (gpu)', flush=True)
        np.save('/mnt/ramdisk/batch.npy', imgs)
        
        # tell gpu that data is ready
        gpu_queue.put(start_index)
    
    gpu_queue.put('stop')


def cpu_worker(df, batch_size, gpu_queue, cpu_queue, mem_queue, cpu_index):
    
    DEM_shape, IR_shape = cpu_queue.get()
    
    #print('got DEM and IR shapes (cpu_{})'.format(cpu_index), flush=True)
    
    for i in range(int(np.ceil(len(df)/batch_size))):
        curr_batch_size = min(batch_size, len(df) - i*batch_size)
        batch = df.iloc[i*batch_size:i*batch_size + curr_batch_size]
        
        start_index = batch.iloc[0].name
        
        #print('making batch (cpu_{})'.format(cpu_index), flush=True)
        coords = make_batch(batch, DEM_shape, IR_shape)
        
        # wait for signal from mem_worker that its safe to save coords
        #print('waiting for signal (cpu_{})'.format(cpu_index), flush=True)
        signal = cpu_queue.get()
        #print('signal received (cpu_{})'.format(cpu_index), flush=True)
        if signal == 'stop':
            return
        
        coords = coords.astype(int)
        np.save('/mnt/ramdisk/coords_{}.npy'.format(cpu_index), coords)
        
        # tell mem_worker and gpu_worker that coords are ready
        #print('sending signal (cpu_{})'.format(cpu_index), flush=True)
        mem_queue.put((cpu_index, start_index))
    
    mem_queue.put('done')

def gpu_worker(df, batch_size, n_cpu_workers, model):
    gpu_queue = Queue()
    mem_queue = Queue()
    cpu_queues = [Queue() for _ in range(n_cpu_workers)]
    gpu_to_mem_queue = Queue()
    
    # splitting data
    # indices must be in order
    split = np.array_split(df, n_cpu_workers)
    
    mem = Process(target=memory_worker, args=(mem_queue, cpu_queues, gpu_to_mem_queue, gpu_queue))
    cpus = [Process(target=cpu_worker, args=(split[i], batch_size, gpu_queue, cpu_queues[i], mem_queue, i))
            for i in range(n_cpu_workers)]
    mem.start()
    for cpu in cpus:
        cpu.start()
    
    # initial signals
    gpu_to_mem_queue.put('ok')
    
    
    while True:
        # wait for signal
        signal = gpu_queue.get()
        if signal == 'stop':
            return
        # otherwise signal is the start index
        start_index = signal
        # load data
        batch = np.load('/mnt/ramdisk/batch.npy')
        
        # tell mem_worker that it's safe to laod more data
        gpu_to_mem_queue.put('ok')
        preds = model.predict(batch)
        df.iloc[start_index:start_index+len(batch), df.columns.get_loc('prob')] = preds
        #print(preds.mean(), start_index)
        print(start_index)
        print('did predictions (gpu)', flush=True)


def metrics(ground_truth, preds, thresh_rad=0.25, thresh_longlat2=0.5, return_value=False):
    matched = metric.kn_match_craters(ground_truth, preds, *cols, thresh_rad=thresh_rad,
                                      thresh_longlat2=thresh_longlat2, radius=cfg.R_planet)[0]
    N_match = len(matched)
    N_ground = len(ground_truth)
    N_detect = len(preds)
    precision = N_match / N_detect
    recall = N_match / N_ground
    fscore = 2 * precision * recall / (precision + recall)
    if return_value:
        return (precision, recall, fscore, N_ground, N_detect, N_match)
    print('Precision: {:1.1f}'.format(100 * precision))
    print('Recall: {:1.1f}'.format(100 * recall))
    print('F1 Score: {:1.1f}'.format(100 * fscore))

# master functions

def data_generation():
    DEM, IR = load_DEM_IR()
    
    craters = load_craters()
    
    for i in range(50):
        start_index = i * 1000
        print('\n{:05d}'.format(start_index), flush=True)
        data.gen_dataset(DEM, IR, craters, 'ran_start_to_finish', start_index, 'random',
                         min_box_size=min_box_size, max_box_size=max_box_size,
                         min_lat=min_lat, max_lat=max_lat, min_long=min_long, max_long=max_long)

def train_post_processing_net():
    DEM, IR = load_DEM_IR()
    
    craters = load_craters()
    
    craters = filter_df(craters, min_lat=min_lat, max_lat=max_lat,
                        min_long=min_long, max_long=max_long,
                        min_diam=min_diam, max_diam=max_diam)

    # Build Model
    n_kernels = 100
    kernel_size = 3
    n_inputs = 2
    lr = 0.00001
    model = ppn.build_model(n_kernels, kernel_size, n_inputs, lr)

    # Train model
    batch_size = 10
    min_d = 0
    max_d = np.inf
    samples_per_epoch = 1000
    epochs = 200
    
    # Callbacks
    now = time.strftime('%c')
    log_dir = os.path.join(cfg.root_dir, 'post_processing_net/logs', now)
    tensorboard = TensorBoard(log_dir, batch_size=batch_size)
    save_folder = os.path.join(cfg.root_dir, 'post_processing_net/models', now)
    os.mkdir(save_folder)
    save_name = os.path.join(save_folder, '{epoch:01d}-{binary_accuracy:.3f}.hdf5')
    save_model = ModelCheckpoint(save_name)
    
    # Train
    print('Beginning training', flush=True)
    model.fit_generator(generator(batch_size, min_d, max_d, DEM, IR, craters),
                        steps_per_epoch=samples_per_epoch//batch_size,
                        epochs=epochs,
                        callbacks=[tensorboard, save_model])
    
def UNET_predictions():
    # Can be re-run if not all iterations are finished successfully
    DEM, IR = load_DEM_IR()
    
    box_sizes = np.exp(np.linspace(np.log(min_box_size), np.log(max_box_size), n_box_sizes))
    print('Box sizes: ', box_sizes.round(1))
    sys_pass = data.systematic_pass(box_sizes, min_lat=min_lat, max_lat=max_lat, min_long=min_long, max_long=max_long)
    n_files = len(sys_pass) // 1000 + 1
    empty_craters = pd.DataFrame(columns=['Long', 'Lat', 'Diameter (km)'])
    
    # Cell can be re-run if there are missing files

    for i in range(n_files):
        start_index = i * 1000
        do_IR = not os.path.isfile('./data/predictions3/IR/sys_moon_craterdist_{:05d}.npy'.format(start_index))
        do_DEM = not os.path.isfile('./data/predictions3/DEM/sys_moon_craterdist_{:05d}.npy'.format(start_index))
    
        # Generate images
        if do_IR or do_DEM:
            print('Making dataset for file {:05d}'.format(start_index), flush=True)
            data.gen_dataset(DEM, IR, empty_craters, 'sys_moon', start_index, 'systematic',
                         sys_pass=sys_pass, in_notebook=False)
    
            if do_IR or do_DEM:
                print('Making Predictions (IR and DEM) {}'.format(start_index), flush=True)
                Popen(["./cnn_ir_dem.bash",str(start_index)])

def filtering():
    box_sizes = np.exp(np.linspace(np.log(min_box_size), np.log(max_box_size), n_box_sizes))
    print('Box sizes: ', box_sizes.round(1))
    sys_pass = data.systematic_pass(box_sizes, min_lat=min_lat, max_lat=max_lat, min_long=min_long, max_long=max_long)
    n_files = len(sys_pass) // 1000 + 1
    
    # Load my crater list (unfiltered IR)
    
    n_files = len(sys_pass) // 1000 + 1
    
    craters_np = np.empty([0,3])
    for i in tqdm(range(n_files)):
        crater_file_name = './data/predictions3/IR/sys_moon_craterdist_{:05d}.npy'.format(i * 1000)
        craters_np = np.vstack([craters_np, np.load(crater_file_name)])
    craters_np[:,2] *= 2 # convert radii to diameters
    my_craters_IR = pd.DataFrame(craters_np, columns=cols)
    
    # Load my crater list (unfiltered DEM)
    
    craters_np = np.empty([0,3])
    for i in tqdm(range(n_files)):
        crater_file_name = './data/predictions3/DEM/sys_moon_craterdist_{:05d}.npy'.format(i * 1000)
        craters_np = np.vstack([craters_np, np.load(crater_file_name)])
    craters_np[:,2] *= 2 # convert radii to diameters
    my_craters_DEM = pd.DataFrame(craters_np, columns=cols)
    
    # Combining lists
    
    my_craters_DEM['DEM'] = 1
    my_craters_DEM['IR'] = 0
    my_craters_IR['DEM'] = 0
    my_craters_IR['IR'] = 1
    
    my_craters_combined = pd.concat([my_craters_DEM, my_craters_IR]).sample(frac=1).copy()
    my_craters_combined = my_craters_combined.reset_index(drop=True)
    
    del my_craters_DEM, my_craters_IR
    
    craters = my_craters_combined
    craters = craters.reset_index(drop=True)
    
    # convert to natural units
    
    rescaled = craters.copy()
    
    rescaled.rename(columns={'Long':'long', 'Lat':'lat', 'Diameter (km)':'d'}, inplace=True)
    
    rescaled['long'] *= np.pi / 180
    rescaled['lat'] *= np.pi / 180
    rescaled['d'] /= cfg.R_planet
    
    
    min_d = rescaled['d'].min()
    max_d = rescaled['d'].max()
    
    # get approximate neighbour lists with generous matching criteria
    
    bin_min = min_d
    
    batch_size = 10_000
    
    neighbours = []
    
    while bin_min < max_d:
        neighbours.append(radius_nbrs(bin_min, rescaled, batch_size))
        bin_min *= (1 + t_rad)
    
    # combine all neighbours into one dictionary
    
    true_neighbours = merge_dicts(neighbours)
    
    craters_np = np.array(craters[cols])
    
    # refine matches with correct criteria
    
    keys = np.array(list(true_neighbours.keys()))
    np.random.shuffle(keys)
    
    for k in tqdm(keys):
        for i in true_neighbours[k]:
            if not is_match(craters_np[k], craters_np[i]):
                true_neighbours[k] = true_neighbours[k][true_neighbours[k] != i]
    filtered = filter_craters(true_neighbours, craters_np)
    filtered = pd.DataFrame(data=filtered, columns=cols) 
    
    # add number of duplicates and the percent of duplicates that come from each list (DEM or IR)
    
    filtered['DEM'] = 0
    filtered['IR'] = 0
    filtered['duplicates'] = 0
    
    filtered_np = np.array(filtered)
    craters_DEM_np = np.array(craters['DEM'])
    
    for i in tqdm(range(len(filtered_np))):
        ns = true_neighbours[i]
        filtered_np[i,3] = craters_DEM_np[ns].mean() #DEM
        filtered_np[i,4] = 1 - filtered_np[i,3] #IR
        filtered_np[i,5] = len(ns) #duplicates
    
    # save results
    
    filtered_2 = pd.DataFrame(filtered_np, columns=['Long', 'Lat', 'Diameter (km)', 'DEM', 'IR', 'duplicates'])
    filtered_2.to_csv('./all_in_one_with_IR_DEM_dup.csv', index=False)


def post_processing_predictions():
    # Load craters
    
    avg_combined_filtered = pd.read_csv('./all_in_one_with_IR_DEM_dup.csv')
    
    # Load post-processing model
    
    model = load_model(cfg.moon_post_processing_model)
    
    avg_combined_filtered['prob'] = None
    batch_size = 100
    n_cpus = 2
    gpu_worker(avg_combined_filtered, batch_size, n_cpus, model)
    avg_combined_filtered.to_csv('./craters_with_probs.csv', index=False)
    
    
def final_selection():
    # Load my craters
    
    avg_combined_filtered = pd.read_csv('./craters_with_probs.csv')
    
    GT_craters = load_craters()
    
    GT_craters_filtered = filter_df(GT_craters, min_lat=min_lat, max_lat=max_lat,
                                    min_long=min_long, max_long=max_long, min_diam=min_diam,
                                    max_diam=max_diam)
    
    avg_combined_filtered = filter_df(avg_combined_filtered, min_lat=min_lat, max_lat=max_lat,
                                      min_long=min_long, max_long=max_long, min_diam=min_diam,
                                      max_diam=max_diam)
    
    # Training set for GBC
    
    n_train = 1000
    train = avg_combined_filtered.sample(n_train).copy()
    matched = metric.kn_match_craters(train, GT_craters_filtered, *cols, radius=cfg.R_planet)[0]
    df = pd.merge(train, matched, how='left', left_on=cols, right_on=['A_' + col for col in cols], indicator='ind')
    train['matched'] = (df['ind'] == 'both').values
    
    # Train GBC
    
    classification_cols = ['duplicates', 'Diameter (km)', 'DEM', 'prob']
    
    clf = GradientBoostingClassifier()
    clf.fit(train[classification_cols], train['matched'])
    
    # GBC predictions
    
    avg_combined_filtered['preds'] = clf.predict(avg_combined_filtered[classification_cols])
    
    # Final predictions
    
    final_predictions = avg_combined_filtered[avg_combined_filtered['preds']].copy()
    final_predictions.to_csv('./final_predictions.csv', index=False)
    
    # Compare against ground truth
    
    metrics(GT_craters_filtered, final_predictions, return_value=False)

if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == 'data_generation':
        data_generation()
    elif arg == 'train_post_processing_net':
        train_post_processing_net()
    elif arg == 'UNET_predictions':
        UNET_predictions()
    elif arg == 'filtering':
        filtering()
    elif arg == 'post_processing_predictions':
        post_processing_predictions()
    elif arg == 'final_selection':
        final_selection()
    else:
        print('Invalid Argument: ', arg)
    