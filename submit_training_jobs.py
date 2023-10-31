import os
import submitit
import train
from collections import OrderedDict
import itertools
import util.util as util
from typing import Any
from options.train_options import TrainOptions


class Trainer:
    def __init__(self, opt) -> None:
        self.opt = util.copyconf(opt)
        
    def __call__(self) -> Any:
        train.main(self.opt)
        
    def checkpoint(self) -> Any:
        import os
        import submitit
        epoch = util.find_latest_checkpoint_epoch(self.opt)
        if epoch is None:
            raise Exception("No checkpoint found")
        self.opt.continue_train = True
        self.opt.epoch_count = epoch + 1
        
        print('Requeueing ', self.opt)
        empty_trainer = type(self)(self.opt)
        return submitit.helpers.DelayedSubmission(empty_trainer)

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
                curr_opts.extend(['--display_id', '0'])
                curr_opts.extend(['--filelist_root', '/gpfs/u/home/LMTM/LMTMsmms/scratch/projects/synthetic-cdm/CDS_pretraining/data'])
                curr_opts.extend(['--dataset', dataset])
                curr_opts.extend(['--source', src])
                curr_opts.extend(['--target', tgt])
                curr_opts.extend(['--train_split', f'train_cls_disjoint{split_src}'])
                curr_opts.extend(['--train_split2', f'train_cls_disjoint{split_tgt}'])
                
                curr_opts.extend(['--name', f'cut_{dataset}_{scenario}_{split_scenario}'])
                curr_opts.extend(['--batch_size', '4'])
                curr_opts.extend(['--dataset_mode', 'unalignedfilelist'])
                
                curr_opts.extend(['--max_dataset_size', '1000'])
                curr_opts.extend(['--save_epoch_freq', '5'])
                curr_opts.extend(['--save_latest_freq', '5000'])
                curr_opts = TrainOptions(cmd_line=' '.join(curr_opts)).parse()
                
                save_dir = os.path.join(curr_opts.checkpoints_dir, curr_opts.name)
                slurm_dir = os.path.join(save_dir, 'slurm')
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