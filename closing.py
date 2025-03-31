from skimage.morphology import erosion, dilation, opening, closing, area_closing, binary_closing
from skimage.morphology import ball, disk
import tifffile as tfl
from skimage.measure import regionprops_table, label, regionprops
import numpy as np
from csbdeep.io import save_tiff_imagej_compatible
from stardist import fill_label_holes
import os
from pathlib import Path
from glob import glob

os.chdir('/mnt/home/ajacinto/ceph/stardist_data/2024-07-15_131317/nuclear_segmentations')
ending = '_closed'
for label_image in glob('*.tif'):
	
	file_path = label_image
	label_image = tfl.imread(label_image)
	label_image = fill_label_holes(label_image)
	print(label_image.shape)
	return_label_image = label_image.copy()
	return_label_image = np.zeros_like(label_image)
	print(np.unique(return_label_image))
	nuclei = np.unique(label_image)
	print(nuclei)
	for nucleus in nuclei:
		if nucleus == 0:
			continue
		print(nucleus)
		individual = label_image == nucleus
		#tfl.imwrite(f'/Users/ajacinto/Desktop/tp_10_corrected_nuclear_seg_{nucleus}.tif',individual)
		individual = binary_closing(individual, ball(3))
		return_label_image[np.where(individual)] = nucleus
	

	save_tiff_imagej_compatible(f'/mnt/home/ajacinto/ceph/stardist_data/2024-07-15_131317/nuclear_segmentations_filled/{Path(file_path).stem}{ending}.tif', return_label_image.astype('uint16'),axes='ZYX')
