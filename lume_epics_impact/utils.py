import toml


def load_configuration(filepath):

    config = toml.load(filepath)

    WORKDIR=config["machine"].get('workdir')
    if not WORKDIR:
        raise ValueError("workdir not defined in toml.")


    HOST = config["machine"].get('host') # mcc-simul or 'sdf'
    if not HOST:
        raise ValueError("host not defined in toml.")

        
    IMPACT_CONFIG_FILE=config["machine"].get('config_file')
    if not IMPACT_CONFIG_FILE:
        raise ValueError("config_file not defined in toml.")

    DISTGEN_INPUT_FILE=config["machine"].get('distgen_input_file')
    if not DISTGEN_INPUT_FILE:
        raise ValueError("distgen_input_file not defined in toml.")
        
    # Directory for summary output
    SUMMARY_OUTPUT_DIR = config["machine"].get('summary_output_dir')
    if not SUMMARY_OUTPUT_DIR:
        raise ValueError("summary_output_dir not defined in toml.")

    # Directory to output plots
    PLOT_OUTPUT_DIR = config["machine"].get('plot_output_dir')
    if not PLOT_OUTPUT_DIR:
        raise ValueError("plot_output_dir not defined in toml.")


    # Directory for archive files
    ARCHIVE_DIR = config["machine"].get('archive_dir')
    if not ARCHIVE_DIR:
        raise ValueError("archive_dir not defined in toml.")

    # Directory for EPICS snapshot files
    SNAPSHOT_DIR = config["machine"].get('snapshot_dir')
    if not SNAPSHOT_DIR:
        raise ValueError("snapshot_dir not defined in toml.")

    # Dummy file for distgen
    DISTGEN_LASER_FILE = config["machine"].get('distgen_laser_file')
    if not DISTGEN_LASER_FILE:
        raise ValueError("distgen_laser_file not defined in toml.")

    # Number of processors
    NUM_PROCS = config["machine"].get('num_procs')
    if not NUM_PROCS:
        raise ValueError("num_procs not defined in toml.")
    else:
        NUM_PROCS = int(NUM_PROCS)


    MPI_RUN_CMD = config["machine"].get("mpi_run_cmd")
    if not MPI_RUN_CMD:
        raise ValueError("mpi_run_cmd not defined in toml.")

    return config