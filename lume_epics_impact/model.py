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
from lume_epics_impact.dashboard import make_dashboard
import pandas as pd
from time import sleep, time
import logging
import os

import numpy as np
import os


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
        self._model_name = configuration["model"].get("model")



        if self._model_name == "cu_inj":
            self._pv_mapping = pd.read_csv(CU_INJ_MAPPING)

        elif self._model_name == "f2e_inj":
            self._pv_mapping = pd.read_csv(F2E_INJ_MAPPING)

        self._pv_mapping.set_index("impact_name")


        self._settings = {
            'distgen:n_particle': self._configuration["model"].get('distgen:n_particle'),   
            'timeout': self._configuration["model"].get('timeout'),
            'header:Nx': self._configuration["model"].get('header:Nx'),
            'header:Ny': self._configuration["model"].get('header:Ny'),
            'header:Nz': self._configuration["model"].get('header:Nz'),
            'numprocs': self._configuration["machine"].get('num_procs'),
            'mpi_run': self._configuration["machine"].get('mpi_run_cmd'),
            'workdir': self._configuration["machine"].get('workdir'),
            'command': self._configuration["machine"].get('command'),
            'command_mpi': self._configuration["machine"].get('command_mpi'),
            'stop': self._configuration["machine"].get('stop'),
            'distgen:t_dist:length:value': self._configuration["distgen"].get('distgen:t_dist:length:value'),
        }

        # Update settings with impact factor
        self._settings.update(dict(zip(self._pv_mapping['impact_name'], self._pv_mapping['impact_factor'])))

        self._archive_dir = self._configuration["machine"].get("archive_path")
        self._plot_dir = self._configuration["machine"].get("plot_output_dir")
        self._summary_dir = self._configuration["machine"].get("summary_output_dir")
        self._distgen_laser_file = self._configuration["distgen"].get("distgen_laser_file")
        self._distgen_input_file = self._configuration["distgen"].get("distgen_input_file")

        self._impact_config = {
            'workdir': self._configuration["machine"].get('workdir'),
            'impact_config': self._configuration["machine"].get('config_file'),
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

        self._settings['distgen:r_dist:file'] = self._distgen_laser_file

        # generate distgen dis
        G = Generator(gfile)

        #G['xy_dist:file'] =  DISTGEN_LASER_FILE #'distgen_laser.txt'

       # G['xy_dist:file'] = self._distgen_laser_file
       # G['n_particle'] = self._configuration["model"]["distgen_n_particle"]
        G.run()
        G.particles.plot('x', 'y', figsize=(5,5))


        dat = {'isotime': itime, 
            'inputs': self._settings, 'config': self._impact_config, 'pv_mapping_dataframe': self._pv_mapping.to_dict()}


        logger.info(f'Running evaluate_impact_with_distgen...')

        t0 = time()

        dat['outputs'] = evaluate_impact_with_distgen(self._settings,
                                            merit_f=lambda x: run_merit(x, itime, self._dashboard_kwargs),
                                            archive_path=self._archive_dir,
                                            **self._impact_config, verbose=False )

        logger.info(f'...finished in {(time()-t0)/60:.1f} min')

        # write summary file
        fname = fname=f'{self._summary_dir}/{self._model_name}-{itime}.json'
        json.dump(dat, open(fname, 'w'))
        #print('Written:', fname)
        logger.info(f'Output written: {fname}')

        for var in self.output_variables:
            var.value = dat["output"][var.name]

        return self.output_variables



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


def run_merit(impact_obj, itime, dashboard_kwargs):
    merit = default_impact_merit(impact_obj)
    # Make the dashboard from the evaluated object
    plot_file = make_dashboard(impact_obj, itime=itime, **dashboard_kwargs)
    #print('Dashboard written:', plot_file)
    logger.info(f'Dashboard written: {plot_file}')
    
    # Assign extra info
    merit['plot_file'] = plot_file    
    merit['isotime'] = itime
    
    # Clear any buffers
    plt.close('all')
    return merit


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

