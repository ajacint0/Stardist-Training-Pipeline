import tifffile as tfl
import numpy as np
from glob import glob
from skimage.measure import regionprops_table, label, regionprops
from skimage.morphology import erosion, dilation, ball, binary_dilation
import os

from tqdm import tqdm

# given an instance segmentation of pole cells that contains small spheres put in place of nuclei during divisions,
# this code turns those small spheres into spheres that match the average size of the rest of the normal nuclei in that segmentation


def createCircularMask(shape, center, radius):

   x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
   distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
   mask = distances <= radius
   return mask


    


os.chdir('/mnt/ceph/users/hnunley/PoleCellProject_Celia/2023-07-12_164901/nuclear_segmentation/')
images = glob('*.tif')
for f in images:

	# collects regular nuclei and the artificial spheres put in and separates them into 2 lists
	
	original_nuclei = []
	small_nuclei = []
	original_labels = []

	print(f)
	img = tfl.imread(f)
	props = regionprops(img)
	for nucleus in props:

		# puts the spacial information of the nucleus in a list so we can easily obtain minimum and maximum values from these lists

		z = []
		y = []
		x = []
		for coord in nucleus.coords:
			z.append(coord[0])
			y.append(coord[1])
			x.append(coord[2])

		diameters = []
		
		# makes sure that each part of the analyzed nucleus is inside the image, then adds the diameter of the nucleus from each plane into a list
		# if that plane of the nucleus is fully in the image

		if (max(z) < img.shape[0] and min(z) > 0):
			diameters.append(max(z) - min(z))
		if (max(y) < img.shape[1] and min(y) > 0):
			diameters.append(max(y) - min(y))
		if (max(x) < img.shape[2] and min(x) > 0):
			diameters.append(max(x) - min(x))
		all_equal = False
		#print(f'diameters for nucleus {nucleus.label} are {diameters}')

		# determines if nucleus is normal or is ann artificial sphere
		# it is an artificial sphere if the diameters of the nucleus in all planes are equal 
		# if one of the diameters cuts out of the image, and if the remaining 2 are equal, it is an artifical sphere

		if len(diameters) == 2:
			if (diameters[0] == diameters[1]):
				all_equal = True
		if len(diameters) == 3:
			if (diameters[0] == diameters[1] and diameters[1] == diameters[2]):
				all_equal = True
		if (all_equal == True):
			small_nuclei.append(nucleus)
			
		else:
			original_labels.append(nucleus.label)
			original_nuclei.append(nucleus)

	# finds the average radius of the normal nuclei
	
	radius_sum = 0		
	for original_nucleus in original_nuclei:
		radius_sum = radius_sum + (original_nucleus.equivalent_diameter_area / 2)

	# if almost all the nuclei are artificial, then simply dilates all nuclei

	if (len(original_nuclei) < 2):
		img = dilation(img, ball(5))
	else:

		# slightly enlarges radius to make a bigger sphere, works better

		avg_radius = radius_sum / len(original_nuclei)
		avg_radius = avg_radius * 1.5
		count = 0
		for small_nucleus in small_nuclei:
			
			# since small nuclei are added after segmentation, they will have a higher value label than the orginal nuclei

			if (small_nucleus.label > max(original_labels)):
				#print(count)
				obj = img == small_nucleus.label
				obj = obj.astype(int)
		
				# deletes artificial nucleus and puts a new bigger one in its place
		
				center = small_nucleus.centroid
				coords = small_nucleus.coords
				img[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
				my_sphere = createCircularMask(obj.shape, center, avg_radius)
				img[my_sphere] = small_nucleus.label
				count = count + 1
	print(f'changed {count} nuclei')
	print()
	tfl.imwrite(f'/mnt/home/ajacinto/ceph/Niles/2023-07-12_164901/resized_nuclear_segmentations/resized_{f}', img)
