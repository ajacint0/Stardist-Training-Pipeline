import tifffile as tfl
from glob import glob
from numpy import random
import numpy as np
import os
from skimage.util import view_as_windows
from pathlib import Path
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import ball, disk
from skimage.segmentation import relabel_sequential
from time import time
from skimage.measure import label
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean
import re

# Code to renumber timepoints when adding 2 timepoint sets together
# target = folder that will have other data added to it
target_folder = '/mnt/home/ajacinto/ceph/training_data/7-12_and_7-15_no_invis/16_bit/raw_7-12/'
new_folder = '/mnt/home/ajacinto/ceph/training_data/7-12_and_7-15_no_invis/16_bit/raw_7-15/'

# Looks at the folder of the training data that will be appended

os.chdir(target_folder)
original_timepoints = []
for original_img in glob('*.tif'):
	original_num = re.findall('\d+', original_img)
	original_num = int(original_num[1])
	original_timepoints.append(original_num)

# gets the greatest timepoint from the original training data folder
# original folder will get stuff added to it

max_original_timepoint = max(original_timepoints)

# Looks at the folder of the training data that will be added

os.chdir(new_folder)
timepoints = []
for img in glob('*.tif'):
	
	num = re.findall('\d+', img)
	num = int(num[1])
	timepoints.append(num)

# Gets the least timepoint from the new training folder

min_timepoint = min(timepoints)

for img in glob('*.tif'):
	
	# replaces timepoint with another based off of the greatest timepoint of the original training folder
	# Original folder will now have timepoints from its old img files + the new ones all in numeric order

	num = re.findall('\d+', img)
	new_img_name = re.sub('p_\d+', 'p_' + str((int(num[1]) - (min_timepoint - 1)) + max_original_timepoint), img, count=1)
	img = tfl.imread(img)
	
	print(f'{target_folder}{new_img_name}')
	tfl.imwrite(f'{target_folder}{new_img_name}', img)
	#np.save(f'{target_folder}{new_img_name}', img)
	
	

