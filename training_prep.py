import numpy as np
import tifffile as tfl
import os
from glob import glob
import re
from csbdeep.utils import Path, normalize
from training_prep_functions import crop_raw, renumber, closing, to_spheres


#raw_paths = ['/mnt/home/ajacinto/ceph/liu_lightsheet_data/nos/2025-06-13_144128_25pct35ms/Raw image/']
#seg_paths = ['/mnt/home/ajacinto/ceph/Niles/2025-06-13_144128_25pct35ms/niles_gui_segmentations/']

#crop_boxes = [[132:564,187:395,120:568],[:432,19:227,41:489],[50:482, 55:263, 90:538],[130:562, :208, 90:538],[:432,:208,:448],[100:532,35:243,117:565],[:0,:0,:0],[:0,:0,:0]]
ch = 0
folder = 'Aggregate_caax'
cropped_paths = []

#for i in range(0, len(paths)):
	#cropped_paths.append(crop_raw(raw_paths[i], folder, crop_boxes[i], ch))
	#crop_seg(seg_paths[i], folder, crop_boxes[i], ch)



cropped_paths_names = ['2023-02-08_150805', '2024-07-25_20pct_50ms', '2025-02-14_113202', '2025-03-11_144214', '2025-03-12_153657', '2025-03-26_144341', '2025-07-09_124552', '2025-07-11_121341', '2025-12-17_160948']

for i in cropped_paths_names:
	cropped_paths.append(f'/mnt/home/ajacinto/ceph/nuclear_segmentation/Aggregate_caax/{i}/raw/')

print(f'cropped_path: {cropped_paths}')



renumber(cropped_paths)
closing(folder)
to_spheres(folder)
