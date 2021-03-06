import toml
from lume_model.models import SurrogateModel
from impact import evaluate_impact_with_distgen
from impact.tools import isotime
from impact.evaluate import  default_impact_merit
import matplotlib.pyplot as plt
from distgen import Generator
import numpy as np
import json
from lume_epics_impact import CU_INJ_MAPPING, F2E_INJ_MAPPING
import pandas as pd
from time import sleep, time
import logging
import os
import yaml
import sys

import numpy as np

from skimage.filters import sobel
from skimage.util import img_as_ubyte
from skimage.segmentation import watershed
from skimage.filters.rank import median
from skimage.morphology import disk


# Gets or creates a logger
logger = logging.getLogger(__name__)  

class ImpactModel(SurrogateModel):
    # move configuration file parsing into utility
    def __init__(self, *, configuration, input_variables, output_variables):

        self.input_variables = input_variables
        self.output_variables = output_variables

        self._configuration = configuration
        self._model_name = configuration["impact"].get("model")

        if self._model_name == "cu_inj":
            self._pv_mapping = pd.read_csv(CU_INJ_MAPPING)

        elif self._model_name == "f2e_inj":
            self._pv_mapping = pd.read_csv(F2E_INJ_MAPPING)

        self._pv_mapping.set_index("impact_name")


        self._settings = {
            'distgen:n_particle': self._configuration["distgen"].get('distgen:n_particle'),   
            'timeout': self._configuration["impact"].get('timeout'),
            'header:Nx': self._configuration["impact"].get('header:Nx'),
            'header:Ny': self._configuration["impact"].get('header:Ny'),
            'header:Nz': self._configuration["impact"].get('header:Nz'),
            'stop': self._configuration["impact"].get('stop'),
            'numprocs': self._configuration["machine"].get('num_procs'),
            'mpi_run': self._configuration["machine"].get('mpi_run_cmd'),
            'workdir': self._configuration["machine"].get('workdir'),
            'command': self._configuration["machine"].get('command'),
            'command_mpi': self._configuration["machine"].get('command_mpi'),
            'distgen:t_dist:length:value': self._configuration["distgen"].get('distgen:t_dist:length:value'),
        }

        # Update settings with impact factor
        self._settings.update(dict(zip(self._pv_mapping['impact_name'], self._pv_mapping['impact_factor'])))

        self._archive_dir = self._configuration["machine"].get("archive_dir")
        self._plot_dir = self._configuration["machine"].get("plot_output_dir")
        self._summary_dir = self._configuration["machine"].get("summary_output_dir")
        self._distgen_laser_file = self._configuration["distgen"].get("distgen_laser_file")
        self._distgen_input_file = self._configuration["distgen"].get("distgen_input_file")


        # kind of odd workaround
        with open(self._configuration["machine"].get('config_file'), "r") as stream:
            try:
                impact_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        impact_config['use_mpi'] = self._configuration["machine"].get('use_mpi')
        impact_config['workdir'] = self._configuration["machine"].get('workdir')
        
        self._impact_config = {
            'workdir': self._configuration["machine"].get('workdir'),
            'impact_config': impact_config,
            'distgen_input_file': self._configuration["distgen"].get('distgen_input_file')
        }
    
        self._dashboard_kwargs = self._configuration.get("dashboard")


    def evaluate(self, input_variables):
        itime = isotime()
        input_variables = {input_var.name: input_var for input_var in input_variables}

        # convert IMAGE vars 
        if input_variables["vcc_array"].value.ptp() < 128:
            downcast = input_variables["vcc_array"].value.astype(np.int8) 
            input_variables["vcc_array"].value = downcast

        if input_variables["vcc_array"].value.ptp() == 0:
            raise ValueError(f'vcc_array has zero extent')

        # scale values by impact factor
        vals = {}
        for var in input_variables.values():
            if var.name in self._pv_mapping["impact_name"]:
                vals[var.name] = var.value * self._pv_mapping.loc[self._pv_mapping["impact_name"] == var.name, "impact_factor"].item()

        # Initialize distgen
        image = input_variables["vcc_array"].value.reshape(input_variables["vcc_size_y"].value, input_variables["vcc_size_x"].value)

        # make units consistent
        if input_variables["vcc_resolution_units"].value == "um/px":
             input_variables["vcc_resolution_units"].value = "um"

        # isolate image
        cutimg = isolate_image(image, fclip=0.08)
    
        assert cutimg.ptp() > 0

        logger.info(f'Initializing distgen...')
        
        write_distgen_xy_dist(self._distgen_laser_file, cutimg, input_variables["vcc_resolution"].value, resolution_units=input_variables["vcc_resolution_units"].value)

        gfile = self._impact_config["distgen_input_file"]

        self._settings['distgen:xy_dist:file'] = self._distgen_laser_file

        df = self._pv_mapping.copy()
        df['pv_value'] = [input_variables[k].value for k in input_variables if "vcc_" not in k]


        dat = {'isotime': itime, 
            'inputs': self._settings, 'config': self._impact_config, 'pv_mapping_dataframe': df.to_dict()}


        logger.info(f'Running evaluate_impact_with_distgen...')

        t0 = time()

        dat['outputs'] = evaluate_impact_with_distgen(self._settings,
                                        #    merit_f=lambda x: run_merit(x, itime, self._dashboard_kwargs),
                                            archive_path=self._archive_dir,
                                            **self._impact_config,
                                            verbose=False)

        logger.info(f'...finished in {(time()-t0)/60:.1f} min')

        for var_name in dat['outputs']:
            if var_name in self.output_variables:
                self.output_variables[var_name].value = dat['outputs'][var_name]

        self.output_variables["isotime"].value = dat["isotime"]

        # write summary file
        fname = fname=f"{self._summary_dir}/{self._model_name}-{dat['isotime']}.json"
        json.dump(dat, open(fname, 'w'))
        logger.info(f'Output written: {fname}')

        return list(self.output_variables.values())



def write_distgen_xy_dist(filename, image, resolution, resolution_units='m'):
    """
    Writes image data in distgen's xy_dist format
    
    Returns the absolute path to the file written
    
    """
    
    # Get width of each dimension
    widths = resolution * np.array(image.shape)
    
    center_y = 0
    center_x = 0
    
    # Form header
    header = f"""x {widths[1]} {center_x} [{resolution_units}]
y {widths[0]} {center_y}  [{resolution_units}]"""
    
    # Save with the correct orientation
    np.savetxt(filename, np.flip(image, axis=0), header=header, comments='')
    
    return os.path.abspath(filename)


def isolate_image(img, fclip=0.08):
    """
    Uses a masking technique to isolate the VCC image
    """
    img=img.copy()
    
    # Clip lowest fclip fraction
    img[img < np.max(img)* fclip] = 0
    
    # Filter out hot pixels to use as a mask
    # https://scikit-image.org/docs/0.12.x/auto_examples/xx_applications/plot_rank_filters.html
    img2 = median(img_as_ubyte(img), disk(2))
    
    elevation_map = sobel(img2)
    markers = np.zeros_like(img2)
    
    # TODO: tweak these numbers
    markers[img2 < .1] = 1
    markers[img2 > .2] = 2

    # Wateshed
    segmentation = watershed(elevation_map, markers)
    
    # Set to zero in original image
    img[np.where(segmentation != 2)]  = 0 
    
    # 
    ixnonzero0 = np.nonzero(np.sum(img2, axis=1))[0]
    ixnonzero1 = np.nonzero(np.sum(img2, axis=0))[0]
    
    i0, i1, j0, j1 = ixnonzero0[0], ixnonzero0[-1], ixnonzero1[0], ixnonzero1[-1]
    cutimg = img[i0:i1,j0:j1]
    
    return cutimg


def isolate_image(img, fclip=0.08):
    """
    Uses a masking technique to isolate the VCC image
    """
    img=img.copy()
    
    # Clip lowest fclip fraction
    img[img < np.max(img)* fclip] = 0
    
    
    # Filter out hot pixels to use aas a mask
    # https://scikit-image.org/docs/0.12.x/auto_examples/xx_applications/plot_rank_filters.html
    img2 = median(img_as_ubyte(img), disk(2))
    
    elevation_map = sobel(img2)
    markers = np.zeros_like(img2)
    
    # TODO: tweak these numbers
    markers[img2 < .1] = 1
    markers[img2 > .2] = 2

    # Wateshed
    segmentation = watershed(elevation_map, markers)
    
    # Set to zero in original image
    img[np.where(segmentation != 2)]  = 0 
    
    # 
    ixnonzero0 = np.nonzero(np.sum(img2, axis=1))[0]
    ixnonzero1 = np.nonzero(np.sum(img2, axis=0))[0]
    
    i0, i1, j0, j1 = ixnonzero0[0], ixnonzero0[-1], ixnonzero1[0], ixnonzero1[-1]
    cutimg = img[i0:i1,j0:j1]
    
    return cutimg

