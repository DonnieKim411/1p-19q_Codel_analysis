import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

def calculate_class_weight(train_label):
	class_weight_dict = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(train_label), train_label)))

	return class_weight_dict