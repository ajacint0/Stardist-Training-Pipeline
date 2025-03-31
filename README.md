# Stardist Training and Evaluation Protocol

### Preparation
This assumes the movie that is going to be analyzed has already been put throught the registration and deconvolution pipeline

## If you have instance segmentation with small added spheres to simulate nuclei during divisions
1. Open resize_nuclei.py
2. Change the directory in the function os.chdir() to the desired location
3. Change the directory in the variable "destination"
4. Change the header for the files that will be written out
5. This Code will transform any small spheres Hayden has put in place of missing nuclei into larger spheres that have the same size as the surrounding nuclei


