import os 
from pathlib import Path
import subprocess

model_id = "BAAI/bge-reranker-v2-m3"
model_vendor, model_name = model_id.split("/")
path_to_model = Path(f"reranker/{model_name}_quantized")
print(path_to_model)
export_command_base = "optimum-cli export openvino --model {} --task text-classification --weight-format int8".format(model_id) + " " + str(path_to_model)
subprocess.run(export_command_base)