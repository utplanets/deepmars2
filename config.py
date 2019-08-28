# size parameters
minrad_ = 5
maxrad_ = 40
#longlat_thresh2_ = 1.8
#rad_thresh_ = 1.0
longlat_thresh2_ = 0.5
rad_thresh_ = 0.25

# These work well for Tues June 4th 10:31
#template_thresh_ = 0.425
#target_thresh_ = 0.3

#template_thresh_ = 0.2
#target_thresh_ = 0.2
#template_thresh_ = 0.75
#target_thresh_ = 0.2

# June 18th
template_thresh_ = 0.475
target_thresh_ = 0.3

root_dir = "/disks/work/james/deepmars2"
DEM_filename = root_dir + "/data/raw/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
IR_filename = root_dir + "/data/raw/Mars_THEMIS_scaled.tif"
crater_filename = root_dir + "/data/raw/RobbinsCraters_20121016.tsv"

# Mars
#R_planet = 3389.5

# Moon
R_planet = 1737.1

# Best pre-trained models
mars_DEM_model = root_dir + '/ResUNET/models/Tue Jun 18 17:26:59 2019/139-0.59.hdf5'
#moon_DEM_model = root_dir + '/ResUNET/models/Fri Jul 19 12:20:22 2019/51-0.60.hdf5'
mars_IR_model = root_dir + '/ResUNET/models/Thu Jun 27 14:04:17 2019/78-0.65.hdf5'
#moon_IR_model = root_dir + '/ResUNET/models/Thu Jul 18 16:31:46 2019/108-0.62.hdf5'
mars_post_processing_model = root_dir + '/post_processing_net/models/Fri Jun 21 09:16:43 2019/61-0.899.hdf5'
#moon_post_processing_model = root_dir + '/post_processing_net/models/Fri Jul 19 10:32:53 2019/17-0.934.hdf5'

# 59m/pix models
#moon_DEM_model = root_dir + '/ResUNET/models/Sat Jul 27 13:15:40 2019/69-0.57.hdf5'
#moon_IR_model = root_dir + '/ResUNET/models/Fri Jul 26 16:40:32 2019/46-0.63.hdf5'

# 59m/pix Robbins trained
moon_DEM_model = root_dir + '/ResUNET/models/Thu Aug  1 17:07:20 2019/77-0.76.hdf5'
moon_IR_model = root_dir + '/ResUNET/models/Fri Aug  2 00:15:49 2019/32-0.81.hdf5'
moon_post_processing_model = root_dir + '/post_processing_net/models/Wed Jul 31 13:36:45 2019/14-0.916.hdf5'
