#%%
from nltools.data import Brain_Data, Design_Matrix
from nltools.stats import zscore
from os.path import isdir, join, isfile, split
import os
import pandas as pd
from bids import BIDSLayout
import time
import json
import sys

class Process():

    def __init__(self, key, process, readable=None, verbose=False):
        self.key = key
        self.process = process
        self.readable = readable or (lambda x: f'{key}-{str(x).replace(" ", "").strip("[]")}')

class Pipeline():

    def __init__(self, name, root):
        
        self.name = name
        self.root = root
        self.layout = BIDSLayout(root, derivatives=True)
        self.tr = self.layout.get_tr()
        with open(self.layout.get(scope=self.name, suffix="pipeline")[0].path) as file:
            pipeline = json.load(file)
        os.chdir(os.path.dirname(self.layout.get(scope=self.name, suffix="pipeline")[0].path))
        self.masks = {mask: Brain_Data(mask_path) for mask, mask_path in pipeline["Masks"].items()}
        for process in pipeline["Processes"]:
            head, tail = os.path.split(os.path.abspath(process["Source"]))
            os.chdir(head)
            print(os.getcwd())
            if tail.endswith(".py"):
                tail = tail[:-3]
            else:
                raise TypeError(f"{tail} is not a Python script.")
            setattr(self, process["Name"], Process(key=process["Readable"], process=getattr(__import__(tail), process["Name"])))


    @staticmethod
    def make_motion_covariates(mc, tr):
        z_mc = zscore(mc)
        all_mc = pd.concat([z_mc, z_mc**2, z_mc.diff(), z_mc.diff()**2], axis=1)
        all_mc.fillna(value=0, inplace=True)
        return Design_Matrix(all_mc, sampling_freq=1/tr)

    
    # def _denoise(sub):
    
    #     print(f"... Smoothing sub {sub}")
        
    #     smoothed = get_smoothed_data(sub=sub)
    #     print(f"... Building designmatrix for sub {sub}")

    #     csf = zscore(pd.DataFrame(smoothed.extract_roi(mask=csf_mask).T, columns=['csf']))

    #     spikes = smoothed.find_spikes(global_spike_cutoff=3, diff_spike_cutoff=3)
    #     covariates = pd.read_csv(layout.get(subject=sub, scope='derivatives', extension='.tsv')[0].path, sep='\t')
    #     mc = covariates[['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z']]
    #     mc_cov = make_motion_covariates(mc, tr)
    #     dm = Design_Matrix(pd.concat([csf, mc_cov, spikes.drop(labels='TR', axis=1)], axis=1), sampling_freq=1/tr)
    #     dm = dm.add_poly(order=2, include_lower=True)

    #     smoothed.X = dm
    #     print(f"... Fitting sub {sub}")
    #     stats = smoothed.regress()

    #     return stats['residual']


    def load_processes(self, sub, return_type="Brain_Data", write="all", **processes):
        if return_type not in ["Brain_Data", "path"]:
            raise Warning(f"Returntype {return_type} not recognised. Returning Brain_Data instance instead.")
            return_type = "Brain_Data"

        path = join(self.root, "derivatives", "python_processing", f"sub-{sub}")

        if not isdir(path):
            raise FileNotFoundError(
                f"Did not find subject {sub} in directory {dir}.")

        for key in processes.keys():
            if key not in self.process_dict:
                raise Warning(f"Key {key} is not understood and thus ignored. Known keys are {self.process_dict.keys()}")
                kwargs.pop("key")

        if len(processes) == 0:
            print("...looking for original file")
            path = self.layout.get(subject=sub, task='localizer', scope='derivatives', suffix='bold', extension='nii.gz', return_type='file')[1]
            if return_type == "path":
                return path
            if return_type == "Brain_Data":
                return Brain_Data(path)

        name = "_".join([f"sub-{sub}", "_".join([f"{self.process_dict[key].key}-{processes[key]}" for key in processes.keys()]), "bold.nii.gz"])

        if isfile(join(path, name)):
            print("...found processed file")
            if return_type == "path":
                return join(path, name)
            data = Brain_Data(join(path, name))
        else:
            yet_to_process = processes.popitem()
            print("...looking with smaller dict")
            data = self.process_dict[yet_to_process[0]].process(
                sub,
                self.load_processes(
                    sub=sub,
                    return_type="Brain_Data",
                    write=("all" if write == "all" else "none"),
                    **processes
                ),
                yet_to_process[1]
            )
            if write in ["all", "main"]:
                print("...writing")
                data.write(join(path, name))
                self.layout = BIDSLayout(self.root, derivatives=True)
        
        if return_type == "Brain_Data":
            return data
        if return_type == "path": 
            return join(dir, f"sub-{sub}", name)
#%%
Pipeline("PsyPypeline", r"C:\Users\hulin\Documents\Uni\20WiSe\Masterarbeit\psypypeline\example")
# %%
