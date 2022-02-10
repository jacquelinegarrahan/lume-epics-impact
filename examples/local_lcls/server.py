from lume_epics.epics_server import Server
from lume_model.utils import variables_from_yaml
from lume_epics.utils import config_from_yaml
import os
from lume_epics_impact.model import ImpactModel
from lume_epics_impact.utils import load_configuration
import logging

if __name__ == "__main__":
  #  with open("examples/local_lcls/variables.yml", "r") as f:

    with open("/Users/jgarra/sandbox/lume-epics-impact/examples/examples/local_lcls/variables.yml", "r") as f:
        input_variables, output_variables = variables_from_yaml(f)

  #  with open("examples/local_lcls/epics_config.yml", "r") as f:
    with open("/Users/jgarra/sandbox/lume-epics-impact/examples/local_lcls/epics_config.yml", "r") as f:

        epics_config = config_from_yaml(f)

    config = load_configuration("/Users/jgarra/sandbox/lume-epics-impact/examples/local_lcls/local_lcls.toml")

    server = Server(
        ImpactModel,
        epics_config,
        model_kwargs={
            "input_variables": input_variables,
            "output_variables": output_variables,
            "configuration": config
        },
    )
    # monitor = False does not loop in main thread
    server.start(monitor=True)
