# Stardist Training and Evaluation Protocol

### Preparation
This assumes the movie that is going to be analyzed has already been put throught the registration and deconvolution pipeline

## If you have instance segmentation with small added spheres to simulate nuclei during divisions
1. Open resize_nuclei.py
2. Change the directory in the function os.chdir() to the desired location
3. Change the directory in the variable "destination"
4. Change the header for the files that will be written out in the variable "header"
5. Run with command 'python resize_nuclei.py'
6. This code will transform any small spheres Hayden has put in place of missing nuclei into larger spheres that have the same size as the surrounding nuclei

## Run closing algorithm to remove holes in middle of segmentations
1. Open closing.py
2. Change the directory in the function os.chdir() to the desired location
3. Change the ending for the files that will be written out in the variable "ending"
4. Run with command 'python closing.py'
5. This code fills in holes left inside the segmentations

## Run algorithm that ensures all segmentations are star-convex
1. Open to_spheres.py
2. Change the directory in the function os.chdir() to the desired location
3. Change the directory in the variable "output"
4. Change the header for the files that will be written out in the variable "header"


