from tqdm import tqdm
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pyarrow import fs
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

os.environ.setdefault("HADOOP_CONF_DIR", "/path/to/hadoop/conf/")
os.environ.setdefault("JAVA_HOME", "/path/to/java")
os.environ.setdefault("HADOOP_HOME", "/path/to/hadoop")
os.environ.setdefault("ARROW_LIBHDFS_DIR", "/path/to/hadoop/lib/")

hadoop_cmd = os.path.join(os.environ["HADOOP_HOME"], "bin", "hadoop")
if os.path.exists(hadoop_cmd):
    os.environ["CLASSPATH"] = subprocess.check_output(
        f"{hadoop_cmd} classpath --glob", shell=True
    ).decode("utf-8")

hdfs = fs.HadoopFileSystem(
    host=os.getenv("HDFS_HOST", "hdfs://your-hdfs-host"),
    port=int(os.getenv("HDFS_PORT", "8020")),
)


import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
