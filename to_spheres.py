import tifffile as tfl
import numpy as np
from glob import glob
from skimage.measure import regionprops_table, label, regionprops
from skimage.morphology import erosion, dilation, ball, binary_dilation
import os
from stardist import relabel_image_stardist3D, Rays_GoldenSpiral, calculate_extents
from stardist import fill_label_holes, random_label_cmap
from stardist.matching import matching_dataset, matching
from tqdm import tqdm
from csbdeep.io import save_tiff_imagej_compatible

def reconstruction_scores(object, n_rays, anisotropy):
    scores = []
    
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)
    Y_reconstructed = relabel_image_stardist3D(object, rays)
    mean_iou = matching_dataset(object, Y_reconstructed, thresh=0, show_progress=False).mean_true_score
    print("abt to exit function")
    return mean_iou

def createCircularMask(shape, center, radius):

   grid = np.ogrid[tuple(slice(dim) for dim in shape)]
    
   # Create an array of coordinates with respect to the center
   coords = np.array([grid[i] - center[i] for i in range(len(shape))])
    
   # Calculate the distance of each point from the center of the sphere
   distances = np.sqrt(np.sum(coords ** 2, axis=0))
    
   # Create a mask based on the distance
   mask = distances <= radius
    
   return mask


os.chdir('/mnt/home/ajacinto/ceph/stardist_data/2024-07-15_131317/nuclear_segmentations_filled/')
output = '/mnt/home/ajacinto/ceph/stardist_data/2024-07-15_131317/nuclear_segmentations_filled_rays/'
Y =  glob('*.tif')
#Y = ['tp_19_corrected_nuclear_seg_closed.tif']
f_count = False
for f in Y:
	print(f'file: {f}')
	if f_count == True:
		break
	img = tfl.imread(f)
	props = regionprops(img)
	count = 0
	for nucleus in props:
		print(f'count: {count}')
		print(f'nucleus: {nucleus.label}')
		if nucleus.label == 0:
			continue
		
		n_rays = 128
		obj = img == nucleus.label
		obj = obj.astype(int)
		#save_tiff_imagej_compatible('/mnt/home/ajacinto/ceph/stardist_data/2023-07-12_164901/corrected_nuclear_segmentations_filled_rays/tp4.tif', obj.astype('uint16'),axes='ZYX')
		score = reconstruction_scores(obj,n_rays, anisotropy=None)
		if score < .7:
			center = nucleus.centroid
			coords = nucleus.coords
			radius = nucleus.equivalent_diameter_area / 2
			print(f'radius: {radius}')
			img[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
			#obj = np.zeros_like(obj)
			#obj[int(center[0]),int(center[1]),int(center[2])] = 1
			sphere = createCircularMask(obj.shape, center, radius)
			img[sphere] = nucleus.label
			
		#print(score)
		#print('hi')
		count = count + 1
	print(output + 'fixed' + f)
	save_tiff_imagej_compatible(output + 'fixed_' + f, img.astype('uint16'),axes='ZYX')
		
