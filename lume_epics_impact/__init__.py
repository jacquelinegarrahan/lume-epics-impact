import logging
import sys

# Gets or creates a logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.INFO)

# define file handler and set formatter
file_handler = logging.FileHandler(f'test.log')
#formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
formatter    = logging.Formatter(fmt="%(asctime)s :  %(name)s : %(message)s ", datefmt="%Y-%m-%dT%H:%M:%S%z")

# Add print to stdout
logger.addHandler(logging.StreamHandler(sys.stdout))
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)


from pkg_resources import resource_filename

CU_INJ_MAPPING = resource_filename(
    "lume_epics_impact.files.pv_mappings", "cu_inj_impact.csv"
)
F2E_INJ_MAPPING = resource_filename(
    "lume_epics_impact.files.pv_mappings", "f2e_inj_impact.csv"
)