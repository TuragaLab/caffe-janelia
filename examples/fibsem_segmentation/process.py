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

# model files
#modelfile = 'net_iter_20000.caffemodel'
modelfile = 'net_iter_72000.caffemodel'
modelproto = 'net_test_big.prototxt'

# Load the datasets
path = '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/'
# Test set
test_dataset = []

test_dataset.append({})
dname = 'tstvol-520-1-h5'
test_dataset[-1]['name'] = dname
h5im = h5py.File(join(path,dname,'img_normalized.h5'),'r')
h5im_n = pygt.normalize(np.asarray(h5im[h5im.keys()[0]]).astype(float32), -1, 1)
test_dataset[-1]['data'] = h5im_n

test_dataset.append({})
dname = 'tstvol-520-2-h5'
test_dataset[-1]['name'] = dname
h5im = h5py.File(join(path,dname,'img_normalized.h5'),'r')
h5im_n = pygt.normalize(np.asarray(h5im[h5im.keys()[0]]).astype(float32), -1, 1)
test_dataset[-1]['data'] = h5im_n



# Set devices
test_device = 2
print('Setting devices...')
pygt.caffe.set_mode_gpu()
pygt.caffe.set_device(test_device)
# pygt.caffe.select_device(test_device, False)

# Load model
print('Loading model...')
net = pygt.caffe.Net(modelproto, modelfile, pygt.caffe.TEST)

# Process
print('Processing ' + str(len(test_dataset)) + ' volume(s)...')
preds = pygt.process(net,test_dataset)
for i in range(len(test_dataset)):
    print('Saving ' + test_dataset[i]['name'])
    h5file = test_dataset[i]['name'] + '.h5'
    outhdf5 = h5py.File(h5file, 'w')
    outdset = outhdf5.create_dataset('main', preds[i].shape, np.float32, data=preds[i])
    outhdf5.close()
  

