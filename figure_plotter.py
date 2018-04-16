from __future__ import print_function
import numpy as np
import math

import matplotlib.pyplot as plt

def plot_acc_loss_vs_epochs(history):
	
	epochs = len(history.epoch)
	fig = plt.figure(1)
	
	# accuracy figure
	plt.subplot(211)

	plt.plot(range(1,epochs+1),history.history['val_acc'],label='validation')
	plt.plot(range(1,epochs+1),history.history['acc'],label='training')
	plt.legend(loc=0)
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.xlim([1,epochs])
#     plt.ylim([0,1])
	plt.grid(True)
	plt.title("Model Accuracy")
	
	# loss figure
	plt.subplot(212)
	
	plt.plot(range(1,epochs+1),history.history['val_loss'],label='validation')
	plt.plot(range(1,epochs+1),history.history['loss'],label='training')
	plt.legend(loc=0)
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.xlim([1,epochs])
#     plt.ylim([0,1])
	plt.grid(True)
	plt.title("Model loss")


	plt.show()
	# fig.savefig('img/'+str(i)+'-accuracy.jpg')
	# plt.close(fig)

fig = plt.figure()
plt.plot(range(1,n_epoch+1),loss_history.lr,'-o',label='learning rate')
plt.xlabel("epoch")
plt.xlim([0,n_epoch+1])
plt.ylabel("learning rate")
plt.legend(loc=0)
plt.grid(True)
plt.title("Learning rate")
plt.show()
fig.savefig('img/3-learning-rate.jpg')
plt.close(fig)


epochs = 40
history = np.load('flair_history.npy').item()


