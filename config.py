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
