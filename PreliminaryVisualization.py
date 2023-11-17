import os
import pandas as pd
import pyreadr
from pathlib import Path

script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, "5v_cleandf 2.rdata")

print("1")
result = pyreadr.read_r(file_path) 
data = result["df"]
data.head()