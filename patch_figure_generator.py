import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing

# This is for figure generator for ppt slide.

# sample patient to be extracted
basedir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/normalized_data/TCGA-HT-7884'
os.chdir(basedir)

desired_size=[142,142]

def zoompad(array, desired_size):
	array = cv2.resize(array,(desired_size[0],desired_size[1]))
	return array
	
os.chdir(basedir)
FLAIR = np.load('FLAIR_normssn4.npy')
T2 = np.load('T2_normssn4.npy')
T1 = np.load('T1_normssn4.npy')
T1post = np.load('T1post_normssn4.npy')
mask = np.load('truth.npy')

# FLAIR = nib.load('flair.nii.gz').get_data()
# T2 = nib.load('t2.nii.gz').get_data()
# T1 = nib.load('t1.nii.gz').get_data()
# T1post = nib.load('t1Gd.nii.gz').get_data()
# mask = nib.load('truth.nii.gz').get_data()

# mask label is organized as following: 1 = non-enhancing, 2 = edema, 4 = enhancing 
mask[mask==2] = 1
mask[mask==4] = 1

FLAIR_m= FLAIR
T2_m= T2
T1_m= T1
T1post_m= T1post

#Find the largest, 75th, and 50th percentile slices in each dimension
x_sum=np.sum(mask,axis=(1,2))
y_sum=np.sum(mask,axis=(0,2))
z_sum=np.sum(mask,axis=(0,1))

#Check if the patient is codel or non-codel (1 or 0). The ratio between codel and non-codel is 13:130
#So, subsample codel cases 10 times more than codel


xp100=np.percentile(x_sum[np.nonzero(x_sum)],100,interpolation='nearest')
yp100=np.percentile(y_sum[np.nonzero(y_sum)],100,interpolation='nearest')
zp100=np.percentile(z_sum[np.nonzero(z_sum)],100,interpolation='nearest')

x_idx = np.argwhere(x_sum==xp100)[0][0]
y_idx = np.argwhere(y_sum==yp100)[0][0]
z_idx = np.argwhere(z_sum==zp100)[0][0]

B = np.argwhere(mask[x_idx])
(xstart_x, ystart_x), (xstop_x, ystop_x) = B.min(0), B.max(0) + 1     
B = np.argwhere(mask[:,y_idx])
(xstart_y, ystart_y), (xstop_y, ystop_y) = B.min(0), B.max(0) + 1     
B = np.argwhere(mask[:,:,z_idx])
(xstart_z, ystart_z), (xstop_z, ystop_z) = B.min(0), B.max(0) + 1


FLAIR_x1 = zoompad(np.asarray(FLAIR_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)

FLAIR_x1_rot = np.rot90(FLAIR_x1)

FLAIR_y1 = zoompad(np.asarray(FLAIR_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)

FLAIR_y1_rot = np.rot90(FLAIR_y1)

FLAIR_z1 = zoompad(np.asarray(FLAIR_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)




T2_x1 = zoompad(np.asarray(T2_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)

T2_x1_rot = np.rot90(T2_x1)

T2_y1 = zoompad(np.asarray(T2_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)

T2_y1_rot = np.rot90(T2_y1)

T2_z1 = zoompad(np.asarray(T2_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)



T1_x1 = zoompad(np.asarray(T1_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)

T1_x1_rot = np.rot90(T1_x1)

T1_y1 = zoompad(np.asarray(T1_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)

T1_y1_rot = np.rot90(T1_y1)

T1_z1 = zoompad(np.asarray(T1_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)



T1post_x1 = zoompad(np.asarray(T1post_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)

T1post_x1_rot = np.rot90(T1post_x1)

T1post_y1 = zoompad(np.asarray(T1post_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)

T1post_y1_rot = np.rot90(T1post_y1)

T1post_z1 = zoompad(np.asarray(T1post_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)

os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/1p_19q/figures')

np.save('FLAIR_x.npy', FLAIR_x1_rot)
np.save('FLAIR_y.npy', FLAIR_y1_rot)
np.save('FLAIR_z.npy', FLAIR_z1)

np.save('T2_x.npy', T2_x1_rot)
np.save('T2_y.npy', T2_y1_rot)
np.save('T2_z.npy', T2_z1)

np.save('T1_x.npy', T1_x1_rot)
np.save('T1_y.npy', T1_y1_rot)
np.save('T1_z.npy', T1_z1)

np.save('T1post_x.npy', T1post_x1_rot)
np.save('T1post_y.npy', T1post_y1_rot)
np.save('T1post_z.npy', T1post_z1)