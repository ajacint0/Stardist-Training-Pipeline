import numpy as np
import tifffile as tfl
import os
from glob import glob
import re
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
import numpy as np
import os
from skimage.util import view_as_windows
from pathlib import Path
from skimage.morphology import erosion, dilation, opening, closing, area_closing, binary_closing, ball, disk
from skimage.segmentation import relabel_sequential
from time import time
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean
from stardist import fill_label_holes

from glob import glob
from skimage.measure import regionprops_table, label, regionprops
from skimage.morphology import erosion, dilation, ball, binary_dilation
import os
from stardist import relabel_image_stardist3D, Rays_GoldenSpiral, calculate_extents
from stardist import fill_label_holes, random_label_cmap
from stardist.matching import matching_dataset, matching
from tqdm import tqdm
from csbdeep.io import save_tiff_imagej_compatible

#source /mnt/home/hnunley/pyenvname_stardist_updated_4x/bin/activate


def crop_raw(img_path, folder, crop_box, ch):
	

	path_arr = img_path.split('/')
	raw_image_ind = path_arr.index('Raw image')
	movie_name = path_arr[raw_image_ind - 1]

	cropped_path = f'/mnt/home/ajacinto/ceph/nuclear_segmentation/{folder}/{movie_name}/raw_cropped'	

	print(img_path)
	print(cropped_path)
	print(crop_box)

	os.chdir(img_path)
	for f in sorted(glob('Recon*.tif')):

		num = re.findall('\d+', f)
		num0 = int(num[0])
		num1 = int(num[1])
		print(num)
		if f'ch_{ch}' in f:
			
			img = tfl.imread(f)
			
			
			img = img[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3], crop_box[4]:crop_box[5]]
			img = img[:img.shape[0]-img.shape[0]%4, :img.shape[1]-img.shape[1]%16, :img.shape[2]-img.shape[2]%16]
			img = normalize(img, 1, 99, axis=None, clip=True)
			img = (img * 255).astype(np.uint8)
			print(img.shape)
			print('Scaling to 255')

			#tfl.imwrite(f'{cropped_paths[i]}/tp_{num0}_ch_{num1}.tif',img)

	return cropped_path


def crop_seg(img_path, folder, crop_box, ch):
	path_arr = img_path.split('/')
	seg_image_ind = path_arr.index('niles_gui_segmentations')
	movie_name = path_arr[seg_image_ind - 1]
	
	cropped_path = f'/mnt/home/ajacinto/ceph/nuclear_segmentation/{folder}/{movie_name}/seg_cropped'

	print(img_path)
	print(cropped_path)
	print(crop_box)

	os.chdir(img_path)
	for f in sorted(glob('*.tif')):
		img = tfl.imread(f)	
		img = img[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3], crop_box[4]:crop_box[5]]
		img = img[:img.shape[0]-img.shape[0]%4, :img.shape[1]-img.shape[1]%16, :img.shape[2]-img.shape[2]%16]
		print(img.shape)

		#tfl.imwrite(f'{cropped_paths[i]}/tp_{num0}_ch_{num1}.tif',img)

def renumber(cropped_paths):
	count = 0
	for i in range(0, len(cropped_paths)):
		raw_folder = cropped_paths[i]
		os.chdir(raw_folder)
		for f_raw in sorted(glob('*.tif')):
			nums = re.findall('\d+', f_raw)
			tp = int(nums[0])
			ch = int(nums[1])
			f_seg = os.path.abspath(f_raw).replace('raw', 'seg')
			print(os.path.abspath(f_raw))
			print(f_seg)
			
			raw_img = tfl.imread(os.path.abspath(f_raw))
			try:
				seg_img = tfl.imread(f_seg)
			except FileNotFoundError:
				f_seg = f_seg.replace('ch_0', 'ch_1')
				seg_img = tfl.imread(f_seg)
				

			tfl.imwrite(f'/mnt/ceph/users/ajacinto/nuclear_segmentation/Aggregate_caax/aggregated_raw_cropped/tp_{count}_ch_0.tif', raw_img)
			tfl.imwrite(f'/mnt/ceph/users/ajacinto/nuclear_segmentation/Aggregate_caax/aggregated_seg_cropped/tp_{count}_ch_0.tif', seg_img)

			count = count + 1


def closing(folder):
	os.chdir(f'/mnt/home/ajacinto/ceph/nuclear_segmentation/{folder}/aggregated_seg_cropped/')
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
		

		save_tiff_imagej_compatible(f'/mnt/home/ajacinto/ceph/nuclear_segmentation/{folder}/aggregated_seg_cropped_closed/{Path(file_path).stem}.tif', return_label_image.astype('uint8'),axes='ZYX')


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
   print(f'coords type: {type(coords)}')
   print(f'center: {len(center)}')
   # Calculate the distance of each point from the center of the sphere
   distances = np.sqrt(np.sum(coords ** 2, axis=0))
    
   # Create a mask based on the distance
   mask = distances <= radius
   print(f'mask: {mask}')
   return mask 


def to_spheres(folder):

	os.chdir(f'/mnt/home/ajacinto/ceph/nuclear_segmentation/{folder}/aggregated_seg_cropped_closed/')
	output = f'/mnt/home/ajacinto/ceph/nuclear_segmentation/{folder}/aggregated_seg_cropped_closed_rays/'
	header = ''
	Y =  glob('*.tif')
	
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
			
			score = reconstruction_scores(obj,n_rays, anisotropy=None)
			if score < .7:
				center = nucleus.centroid
				coords = nucleus.coords
				radius = nucleus.equivalent_diameter_area / 2
				print(f'radius: {radius}')
				img[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
				
				sphere = createCircularMask(obj.shape, center, radius)
				img[sphere] = nucleus.label
				
			
			count = count + 1
		print(output + 'fixed' + f)
		save_tiff_imagej_compatible(output + header + f, img.astype('uint8'),axes='ZYX')



