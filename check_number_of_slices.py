import os
import numpy as np

basedir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/normalized_data/'
os.chdir(basedir)
patients=next(os.walk('.'))[1]

x_store = np.zeros(len(patients))
y_store = np.zeros(len(patients))
z_store = np.zeros(len(patients))

x_store_50 = np.zeros(len(patients))
y_store_50 = np.zeros(len(patients))
z_store_50 = np.zeros(len(patients))

# check the number of slices if codel

xl = pd.ExcelFile('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/clinical_data_IDH1_1p19q.xlsx')
df = np.asarray(xl.parse("KPS_80"))

patient_1p19q = df[:,[0,3]] #obtain patient id and the 1p19q status

# obtain the ratio between codel and non_codel
num_codel = np.count_nonzero(patient_1p19q[:,1])

codel_slices = np.ndarray([num_codel,4])


codel_counter = 0
for p in range(len(patients)):
	
	print(p, patients[p])
	patient_dir = basedir + patients[p] + '/'

	os.chdir(patient_dir)

	mask = np.load('truth.npy')

	# mask label is organized as following: 1 = non-enhancing, 2 = edema, 4 = enhancing 
	mask[mask==2] = 1
	mask[mask==4] = 1
   
	#Find the largest, 75th, and 50th percentile slices in each dimension
	x_sum=np.sum(mask,axis=(1,2))
	y_sum=np.sum(mask,axis=(0,2))
	z_sum=np.sum(mask,axis=(0,1))

	x_vec = x_sum[np.nonzero(x_sum)]
	x_store[p] = len(x_vec) # store the number of pixels
	x_store_50[p] = np.count_nonzero(x_vec>np.percentile(x_sum[np.nonzero(x_sum)],50,interpolation='nearest')) #find slices that are bigger than 50th percentile

	y_vec = y_sum[np.nonzero(y_sum)]
	y_store[p] = len(y_vec)
	y_store_50[p] = np.count_nonzero(y_vec>np.percentile(y_sum[np.nonzero(y_sum)],50,interpolation='nearest'))

	z_vec = z_sum[np.nonzero(z_sum)]
	z_store[p] = len(z_vec)
	z_store_50[p] = np.count_nonzero(z_vec>np.percentile(z_sum[np.nonzero(z_sum)],50,interpolation='nearest'))
	
	idx_1p19q=np.asarray(np.where((patient_1p19q[:,0].astype(str))==str(patients[p])))
	curr_1p19q = patient_1p19q[idx_1p19q,1]

	if curr_1p19q == 1:
		codel_slices[codel_counter] = np.stack( (idx_1p19q[0][0], len(x_vec), len(y_vec), len(z_vec) ) )
		codel_counter +=1


print(min(x_store))
print(min(y_store))
print(min(z_store))

# based on this analysis, just import upto 30 slices regardless for codel case and 3 slices for non-codel




