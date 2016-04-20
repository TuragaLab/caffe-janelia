from __future__ import print_function
import sys, os, math
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join

# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'
sys.path.append(pygt_path)
import PyGreentea as pygt

# Load the datasets
path = '/data-ssd/john/fly-light-atlas/affineExps/trainingData/'

imname='I3-A6Affine_trainingDat-Image.h5'
xfmname='I3-A6Affine_trainingDat-Transform.h5'

# Train set
train_dataset = []
train_dataset.append({})
train_dataset[-1]['name'] = path
train_dataset[-1]['data'] = np.transpose( np.array(h5py.File(join(path,imname),'r')['main'],dtype=np.float32), [2,1,0])[None,:]
train_dataset[-1]['label'] = np.transpose( np.array(h5py.File(join(path,xfmname),'r')['main'],dtype=np.float32), [3, 2, 1, 0])
train_dataset[-1]['transform'] = {}
train_dataset[-1]['transform']['scale'] = (0.8,1.2)
train_dataset[-1]['transform']['shift'] = (-0.2,0.2)

print('Training set contains ' + str(len(train_dataset)) + ' volumes')
for iset in range(len(train_dataset)):
    print(train_dataset[iset]['name'] + str(iset) + ' shape:' + str(train_dataset[iset]['data'].shape))
    print(train_dataset[iset]['name'] + str(iset) + ' shape:' + str(train_dataset[iset]['label'].shape))

# Train set
#test_dataset = []
#for dname in ['tstvol-520-2-h5','tstvol-520-1-h5']:
#	test_dataset.append({})
#	test_dataset[-1]['name'] = dname
#	test_dataset[-1]['data'] = np.array(h5py.File(join(path,dname,'im_uint8.h5'),'r')['main'],dtype=np.float32)/(2.**8)

#for iset in range(len(train_dataset)):
##    test_dataset[iset]['data'] = test_dataset[iset]['data'][None,:]
#    print(train_dataset[iset]['name'] + str(iset) + ' shape:' + str(test_dataset[iset]['data'].shape))
#    print(train_dataset[iset]['name'] + str(iset) + ' shape:' + str(test_dataset[iset]['label'].shape))

# Set train options
class TrainOptions:
    loss_function = "euclid"
    loss_output_file = "log/loss.log"
    test_output_file = "log/test.log"
    test_interval = 4000
    scale_error = True
    training_method = "affinity"
    recompute_affinity = True
    train_device = 0
    test_device = 2
    test_net=None
    max_iter = int(1e4)
    snapshot = int(2e3)
    loss_snapshot = int(2e3)
    snapshot_prefix = 'net'


options = TrainOptions()

# Set solver options
print('Initializing solver...')
solver_config = pygt.caffe.SolverParameter()
solver_config.train_net = 'net_train_euclid.prototxt'

solver_config.type = 'Adam'
solver_config.base_lr = 1e-8
solver_config.momentum = 0.99
solver_config.momentum2 = 0.999
solver_config.delta = 1e-8
solver_config.weight_decay = 0.000005
solver_config.lr_policy = 'inv'
solver_config.gamma = 0.0001
solver_config.power = 0.75

solver_config.max_iter = options.max_iter
solver_config.snapshot = options.snapshot
solver_config.snapshot_prefix = options.snapshot_prefix
solver_config.display = 1

# Set devices
print('Setting devices...')
pygt.caffe.enumerate_devices(False)
# pygt.caffe.set_devices((options.train_device, options.test_device))
pygt.caffe.set_devices(tuple(set((options.train_device, options.test_device))))

# First training method
solverstates = pygt.getSolverStates(solver_config.snapshot_prefix);
if (len(solverstates) == 0 or solverstates[-1][0] < solver_config.max_iter):
    solver, test_net = pygt.init_solver(solver_config, options)
    if (len(solverstates) > 0):
        solver.restore(solverstates[-1][1])
    pygt.train(solver, None, train_dataset, [], options)
