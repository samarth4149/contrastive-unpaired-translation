import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import numpy as np
import time

opt = TestOptions(
    cmd_line=' '.join([
        '--name', 'cut_domainnet_c2p',
        # '--filelist_root', '/projectnb/ivc-ml/samarth/projects/synthetic/synthetic-cdm/CDS_pretraining/data/',
        '--filelist_root', '/gpfs/u/home/LMTM/LMTMsmms/scratch/projects/synthetic-cdm/CDS_pretraining/data/',
        '--dataset_mode', 'unalignedfilelist',
        '--train_split', 'train_cls_disjoint1',
        '--train_split2', 'train_cls_disjoint2',
        '--dataset', 'domainnet',
        '--phase', 'test',
        '--source', 'clipart',
        '--target', 'painting',
    ])
).parse()

# opt.name = 'cut_domainnet_c2p'
# opt.filelist_root = '/projectnb/ivc-ml/samarth/projects/synthetic/synthetic-cdm/CDS_pretraining/data/'
# opt.dataset_mode = 'unalignedfilelist'
# opt.train_split = 'train_cls_disjoint1'
# opt.train_split2 = 'train_cls_disjoint2'
# opt.dataset = 'domainnet'
# opt.source = 'clipart'
# opt.target = 'painting'

# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

opt.serial_batches = False # False currently to test different images  
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
# train_dataset = create_dataset(util.copyconf(opt, phase="train"))
model = create_model(opt)      # create a model given opt.model and other options

# disable data shuffling; comment this line if results on randomly chosen images are needed.

i = 0
start = time.time()
data = next(iter(dataset.dataloader))  # get the first batch
print('loaded data. time taken: ', time.time() - start)
if i == 0:
    model.data_dependent_initialize(data)
    print('initialized model. cumulative time taken: ', time.time() - start)
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    print('model setup done. cumulative time taken: ', time.time() - start)
    model.parallelize()
    print('model parallelization done. cumulative time taken: ', time.time() - start)
    if opt.eval:
        model.eval()
model.set_input(data)  # unpack data from data loader
model.test()           # run inference
print('Testing done. cumulative time taken: ', time.time() - start)
visuals = model.get_current_visuals()  # get image results
img_path = model.get_image_paths()     # get image paths
import ipdb; ipdb.set_trace()