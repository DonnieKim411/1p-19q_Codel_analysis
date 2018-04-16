# 1p-19q_Codel_analysis

This is a masters thesis work for predicting 1p/19q co-deletion status using BRATS 2017 data which is derived from TCGA/TCIA.

For the image data, every patient image has 4 sequences, T1, T1post, T2, and FLAIR with a mask depicting enhancing, edema, and necrotic/non-enhancing.

It is already skull stripped, co-registered among 4 modalities, and interpolated into 1x1x1 mm.

For 1p/19q codeletion status, I had to find them manually via CBioPortal.

For Preprocessing steps, I performed N4 correction on T1,T1post, and T2 (FLAIR was excluded due to too much reduction in signal). 

Then tumor region intensities were normalized by subtracting median intensity of normal brain region, then dividing by the interquartile intensity of normal brain.

Once normalized, a smallest bounding rectangle was draw around the tumor to capture the tumor region, then the rectangle was resized into 142 x 142 pixels.

This patch extraction was done in all 3 directions: sagittal, coronal, and axial.

The ratio of codeleted vs non-codeleted was 1:10, hence I extracted 2 patches for non-codel (100% and 75% percentile mask area) and 20 patches for codel (top 20 slices in terms of mask area) such that the label ratio is in 50/50.

The data split was done into 70/10/20 with stratified random shuffling to keep the label ratio among 3 data splits.

Training was done individually for each modality by the means of transfer learning using ResNet-34. I retrianed the top 2 conv layers + Fully connected layer (+ drop out (75%) that I manually introduced to overcome over-fitting).

Every modality was trained for 40 epochs with learning rate dropping by 25% for every 10th epoch. 

Also, aggresive data augmentation was used throughout the training: rotation by 30 degrees, shift both horizontally and vertically, shearing, zooming in/out, horizontal/vertical flip.

