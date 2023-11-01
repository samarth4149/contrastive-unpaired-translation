import os
import submitit
import inference
from collections import OrderedDict
import itertools
import util.util as util
from typing import Any
from options.test_options import TestOptions
import sys


class Trainer:
    def __init__(self, opt) -> None:
        self.opt = util.copyconf(opt)
        
    def __call__(self) -> Any:
        inference.main(self.opt)
        
if __name__ == '__main__':
    datasets = OrderedDict({
        'cub': {
            'domains' : ['Real', 'Painting']
        },
        'domainnet': {
            'domains' : ['clipart', 'painting', 'sketch'],
        },
    })
    
    for dataset in datasets:
        for src, tgt in itertools.permutations(datasets[dataset]['domains'], 2):
            for split_src, split_tgt in [(1,2), (2,1)]:
                scenario = f'{src[0]}2{tgt[0]}'
                split_scenario = f'split{split_src}{split_tgt}'
                curr_opts = []
                curr_opts.extend(['--filelist_root', '/gpfs/u/home/LMTM/LMTMsmms/scratch/projects/synthetic-cdm/CDS_pretraining/data'])
                curr_opts.extend(['--dataset', dataset])
                curr_opts.extend(['--source', src])
                curr_opts.extend(['--target', tgt])
                curr_opts.extend(['--train_split', f'train_cls_disjoint{split_src}'])
                curr_opts.extend(['--train_split2', f'train_cls_disjoint{split_tgt}'])
                curr_opts.extend(['--dataset_mode', 'unalignedfilelist'])
                
                name = f'cut_{dataset}_{scenario}_{split_scenario}'
                expt_name = f'inference_{name}'
                root_dir = f'/gpfs/u/home/LMTM/LMTMsmms/scratch/data/synthetic_cdm/synthetic-data/cut/train/{dataset}/{scenario}'
                curr_opts.extend(['--name', name]) # to load latest checkpoint from here
                curr_opts.extend(['--root_dir', root_dir])
                
                curr_opts = TestOptions(cmd_line=' '.join(curr_opts)).parse()
                
                slurm_dir = os.path.join(root_dir, 'slurm', split_scenario)
                os.makedirs(slurm_dir, exist_ok=True)
                
                trainer = Trainer(curr_opts)
                
                num_gpus_per_node = 1
                executor = submitit.AutoExecutor(folder=slurm_dir, slurm_max_num_timeout=30)
                addnl_params = {
                    'gres': f'gpu:{num_gpus_per_node}',
                    'mail_type': 'FAIL',
                    'mail_user' : 'samarthm@bu.edu',
                }
                executor.update_parameters(
                    name=curr_opts.name,
                    mem_gb=20,
                    tasks_per_node=num_gpus_per_node,
                    cpus_per_task=10,
                    timeout_min=360,
                    # slurm_partition='el8',
                    slurm_signal_delay_s=120,
                    slurm_additional_parameters=addnl_params,
                )
                
                print('Expt Name :', curr_opts.name)
                # loop for submitting
                import time
                submitted = False
                num_tries = 0
                while not submitted and num_tries < 10:
                    try:
                        job = executor.submit(trainer)
                        submitted = True
                    except Exception:
                        print('Submission failed. Trying again in 5 seconds')
                        time.sleep(5)
                        submitted = False
                        num_tries += 1
                if num_tries >= 10:
                    raise Exception(f'Failed to submit job {curr_opts.name}')
                
                print("Submitted job_id:", job.job_id)
                
                