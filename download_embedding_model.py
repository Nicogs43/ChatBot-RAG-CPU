import os 
from pathlib import Path
import subprocess

model_id = "BAAI/bge-small-en-v1.5"
path_to_model = Path(f"models/{model_id}")
print(path_to_model)

export_command_base = "optimum-cli export openvino --model {} --task feature-extraction".format(model_id) + " " + str(path_to_model)
subprocess.run(export_command_base)

