# Stardist Training and Evaluation Protocol

### Preparation
This assumes the movie that is going to be analyzed has already been put throught the registration and deconvolution pipeline

## Crop and change dtype of images to 16bit
- The deconvolved images will most likely be too large to put in the training data, only crop the area with the pole cells

## Create movie folders
1. Go to the nuclear_segmentation directory, if it doesn't exist, create one
2. In this directory, create a folder with the name of your training data i.e. 'caax_training'
3. Inside the training data directory, create folders with names corresponding to the pole cell movies you will be training the model on
4. Inside the movie folders create folders 'raw' and 'seg' for each movie, these will contain the raw data and segmentations respectively
5. Open the file 'training_prep.py'
   - set 'user' variable to the name of your account name in ceph i.e. /mnt/ceph/users/```ajacinto```/nuclear_segmentation/caax_training/
   - set 'folder' variable to name of the training set i.e. /mnt/ceph/users/ajacinto/nuclear_segmentation/```caax_training```/
   - set strings in 'cropped_paths_names' list to the names of your movies that you added in step 2
6. Run code by typing 'python training_prep.py' in terminal
   - training_prep.py calls functions from training_prep_functions
      1. renumber() combines all raw data and all segmentations into individual folders
      2. closing() fills all holes left inside the segmentations
      3. to_spheres() changes any non-convex-star shaped segmentation up until this point into spheres
      4. tif_to_npy() changes all the images from .tif to .npy so Andrew's model can use it
      5. split_train_val() separates the data into training and validation bins
   - These functions also create folders which you can check the intermediate contents of
   - example structure for directory will be in '/mnt/home/ajacinto/ceph/nuclear_segmentation/test/'

## Training
1. Open train_convnext_unet_large.yaml
   - Change 'name' parameter to name of the training set with date i.e. 'caax_training_0721_2025'
   - Make sure 'train_image_paths', 'train_mask_paths', 'val_image_paths', and 'val_mask_paths' paths are corect
2. Open run_training.sh
   - In the last line of the file, ensure the path leads to the .yaml file.
  
## Evaluation
1. In training set folder, create folder with name of movie you wish to evaluate on
   - These images should be cropped 16/8bit tifs
2. Go to the jupyter hub from the flatiron website
   - Set job to gpu node (4 hours)
3. Change 'source_dir' variable path to the one you chose for evaluation
4. Change 'config_file' variable path to the path for the .yaml file
5. For the path in the model.net.load_state_dict() function, make sure it points to your checkpoints, probably only have to change name of the training, the same as in the .yaml file.
6. 
     

