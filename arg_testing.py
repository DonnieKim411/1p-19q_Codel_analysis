import os
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

import argparse



if __name__="__main":
	argu = argparse.ArgumentParser()

	argu.add_argument("--train_dir")
	a.add_argument("--val_dir")
	a.add_argument("--nb_epoch", default=NB_EPOCHS)
	a.add_argument("--batch_size", default=BAT_SIZE)
	a.add_argument("--output_model_file", default="inceptionv3-ft.model")
	a.add_argument("--plot", action="store_true")

	args = a.parse_args()
	if args.train_dir is None or args.val_dir is None:
	a.print_help()
	sys.exit(1)

	if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
	print("directories do not exist")
	sys.exit(1)