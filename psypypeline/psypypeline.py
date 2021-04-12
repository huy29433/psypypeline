import json
import os
import sys
from os.path import isdir, isfile, join

import bids
import pandas as pd
from bids import BIDSLayout
from nltools.data import Brain_Data, Design_Matrix
from nltools.stats import zscore


def _get_unique(layout, *args, **kwargs):
    """Helper function to get a unique result instead of a list when calling BIDSLayout.get
    Lets you choose from a list if ambiguous and raises an error if no match is found.

    Parameters
    ----------
    layout : BIDSLayout
        So this can beused as an instance method 

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    FileNotFoundError
        [description]
    """
    results = layout.get(*args, **kwargs)
    if len(results) == 1:
        return results[0]
    if len(results) == 0:
        raise FileNotFoundError(
            f"Did not find a file with the following specifications: {args}, {kwargs}")
    else:
        print(
            f"Found the following files with the specifications {args}, {kwargs}")
        print(*[f"{i:2}: {file}" for i, file in enumerate(results)], sep="\n")
        i = int(input("Enter the number of the correct file: "))
        return results[i]


BIDSLayout._get_unique = _get_unique


class Process():

    def __init__(self, key, process, readable=None, verbose=False):
        self.key = key
        self.process = process
        self.readable = readable or (
            lambda args: Process._readable(self.key, args))

    @staticmethod
    def _readable(key: str, args) -> str:
        """Turns a key and its argument into a readable bids-compatible filename-chunk.

        Parameters
        ----------
        key : str
            Keyword to appear in filename (so e.g. "denoised" and not "denoise)
        args : 
            Argument(s) to appear in filename. May be dict or primitive type.

        Returns
        -------
        str
            filename chunk of the form "key", "key-arg" or "key-{kw-arg, ..., kw-arg}"
        """
        if isinstance(args, dict):
            if args == {}:
                return key
            else:
                return f"{key}-" + str(args).replace("\'", "").replace("\"", "").replace(" ", "").replace(":", "-")
        else:
            return f"{key}-{args}"


class Pipeline():

    def __init__(self, name, root):

        self.name = name
        self.root = root
        self.reload()

    @staticmethod
    def original_path(layout, sub):
        return layout._get_unique(subject=sub, scope="fMRIPrep", suffix='bold', extension='nii.gz', return_type='file')

    def reload(self):
        self.layout = BIDSLayout(self.root, derivatives=True)
        self.tr = self.layout.get_tr()
        with open(self.layout._get_unique(scope=self.name, suffix="pipeline").path) as file:
            pipeline = json.load(file)
        os.chdir(os.path.dirname(self.layout._get_unique(
            scope=self.name, suffix="pipeline").path))
        self.masks = {mask: Brain_Data(mask_path)
                      for mask, mask_path in pipeline["Masks"].items()}

        # Set up the process dictionary

        self.processes = dict()
        for process in pipeline["Processes"]:
            head, tail = os.path.split(os.path.abspath(process["Source"]))
            if tail.endswith(".py"):
                tail = tail[:-3]
            else:
                raise TypeError(f"{tail} is not a Python script.")
            sys.path.append(head)
            self.processes[process["Name"]] = Process(
                key=process["Readable"], process=getattr(__import__(tail), process["Name"]))
            sys.path.remove(head)

    @staticmethod
    def make_motion_covariates(mc, tr):
        z_mc = zscore(mc)
        all_mc = pd.concat(
            [z_mc, z_mc**2, z_mc.diff(), z_mc.diff()**2], axis=1)
        all_mc.fillna(value=0, inplace=True)
        return Design_Matrix(all_mc, sampling_freq=1/tr)

    def load_data(self, sub, return_type="Brain_Data", write="all", verbose=True, **processes) -> Brain_Data:

        #  We'll use this function to print only when being verbose
        def v_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        return_types = ["Brain_Data", "path"]

        if return_type not in return_types:
            raise TypeError(
                f"Returntype {return_type} not recognised. Must be in {return_types}.")

        path = join(self.root, "derivatives", self.name, f"sub-{sub}")

        if not isdir(path):
            raise FileNotFoundError(
                f"Did not find subject {sub} in directory {dir}.")

        for key in processes.keys():
            if key not in self.processes.keys():
                raise KeyError(
                    f"{key} is not known process. Known processes are {self.processes.keys()}")

        if len(processes) == 0:
            v_print(f"...loading the unprocessed file of subject {sub}")
            path = self.original_path(layout=self.layout, sub=sub)
            if return_type == "path":
                return path
            if return_type == "Brain_Data":
                return Brain_Data(path)

        # This is the most important part:
        name = "_".join([f"sub-{sub}", "_".join(
            [f"{self.processes[key].readable(args)}" for key, args in processes.items()]), "bold.nii.gz"])

        if isfile(join(path, name)):
            v_print(f"...found {name}")
            if return_type == "path":
                return join(path, name)
            data = Brain_Data(join(path, name))
        else:
            last_process, last_key = processes.popitem()
            v_print(f"...{name} does not exist yet")
            yet_to_process = self.load_data(
                sub=sub,
                return_type="Brain_Data",
                write=("all" if write == "all" else "none"),
                **processes
            )
            v_print(f"Applying process {last_process}")
            data = self.processes[last_process].process(
                self,
                sub,
                yet_to_process,
                **last_key if isinstance(last_key, dict) else last_key
            )
            if write in ["all", "main"]:
                v_print(f"...writing {name}")
                data.write(join(path, name))
                self.layout = BIDSLayout(self.root, derivatives=True)

        if return_type == "Brain_Data":
            return data
        if return_type == "path":
            return join(dir, f"sub-{sub}", name)

    def create_subject_folders(self, subs, format="sub-{sub}"):
        for sub in subs:
            os.mkdir(join(self.root, "derivatives",
                          self.name, format.format(sub=sub)))

    @staticmethod
    def create(name, root, create_subs=False):

        pipeline_root = join(root, "derivatives", name)

        # create folder
        os.makedirs(pipeline_root, exist_ok=True)

        # create pipeline.json
        with open(join(pipeline_root, "pipeline.json"), "x") as file:
            json.dump({"Name": name, "Processes": [],
                       "Masks": dict()}, file, indent=4)

        # create dataset_description.json
        with open(join(pipeline_root, "dataset_description.json"), "x") as file:
            json.dump(
                {
                    "Name": name,
                    "BIDSVersion": bids.__version__,
                    "PipelineDescription": {
                        "Name": name,
                        "Version": "",
                        "CodeURL": ""
                    },
                    "CodeURL": "",
                    "HowToAcknowledge": ""
                },
                file,
                indent=4
            )

        pipeline = Pipeline(name, root)

        if create_subs:
            pipeline.create_subject_folders(pipeline.layout.get_subjects())

        return pipeline
