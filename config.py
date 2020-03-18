# parameters
minrad_ = 5
maxrad_ = 40
longlat_thresh2_ = 0.5
rad_thresh_ = 0.25
template_thresh_ = 0.475
target_thresh_ = 0.3

root_dir = "/disks/work/lee/dm2/craters"
DEM_filename = root_dir + "/data/raw/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
IR_filename = root_dir + "/data/raw/Mars_THEMIS_scaled.tif"
crater_filename = root_dir + "/data/raw/RobbinsCraters_20121016.tsv"

# Mars
R_planet = 3389.5

# Moon
#R_planet = 1737.1

# Best pre-trained models
# Mars Robbins Trained
mars_DEM_model = root_dir + '/data/ResUNET/models/james_best/mars_DEM_139_0.59.hdf5'
mars_IR_model = root_dir + '/data/ResUNET/models/james_best/mars_IR_78_0.65.hdf5'
mars_post_processing_model = root_dir + '/data/post_processing_net/models/james_best/mars_post_61_0.899.hdf5'

# Moon H&P Trained on 118/100m/pix
#moon_DEM_model = root_dir + '/ResUNET/models/Fri Jul 19 12:20:22 2019/51-0.60.hdf5'
#moon_IR_model = root_dir + '/ResUNET/models/Thu Jul 18 16:31:46 2019/108-0.62.hdf5'
#moon_post_processing_model = root_dir + '/post_processing_net/models/Fri Jul 19 10:32:53 2019/17-0.934.hdf5'

# Moon Robbins Trained on 59m/pix
#moon_DEM_model = root_dir + '/ResUNET/models/Thu Aug  1 17:07:20 2019/77-0.76.hdf5'
#moon_IR_model = root_dir + '/ResUNET/models/Fri Aug  2 00:15:49 2019/32-0.81.hdf5'
#moon_post_processing_model = root_dir + '/post_processing_net/models/Wed Jul 31 13:36:45 2019/14-0.916.hdf5'

