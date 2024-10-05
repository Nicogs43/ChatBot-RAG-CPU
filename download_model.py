from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 
from pathlib import Path
import subprocess
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def convert_to_int8():
    if (int8_model_dir / "openvino_model.xml").exists():
        return
    int8_model_dir.mkdir(parents=True, exist_ok=True)
    #remote_code = model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int8".format(hf_model_id)
    export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int8_model_dir)
    logging.info(export_command)
    subprocess.run(export_command)

def convert_to_int4( int4_mode:str = "SYM", group_size:int = 128, ratio:float = 0.8):
    if (int4_model_dir / "openvino_model.xml").exists():
        return
    #remote_code = model_configuration.get("remote_code", False)
    int4_model_dir.mkdir(parents=True, exist_ok=True)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int4".format(hf_model_id)
    int4_compression_args = " --group-size {} --ratio {}".format(group_size, ratio)
    if int4_mode == "SYM":
        int4_compression_args += " --sym"
    else:
        int4_compression_args += " --asym"
    #if enable_awq.value:
    #   int4_compression_args += " --awq --dataset wikitext2 --num-samples 128"
    export_command_base += int4_compression_args
    #if remote_code:
    export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int4_model_dir)
    logging.info(export_command)
    subprocess.run(export_command)


#insert the model that you want to download
#if it is from OpenVINO, it will be downloaded directly in IR format
#otherwise, it will be downloaded in PyTorch format and then converted to IR format
#model_id = "OpenVINO/Phi-3-mini-128k-instruct-int8-ov"

hf_model_id = input("Insert the model id in HF (model_vendor/actual_model_id): ")
#check if the model name is insert in the correct format (model_vendor/actual_model_id)
if hf_model_id.count("/") != 1:
    logging.info("The model name must be in the format model_vendor/actual_model_id")
    sys.exit()
#split the model_id in model_vendor and actual_model_id
model_vendor, model_id = hf_model_id.split("/")
local_model_path = f'./model/{model_vendor}/{model_id}'
os.makedirs(local_model_path, exist_ok=True)
logging.info(f"Downloading the model {model_id} from {model_vendor}")

if model_vendor == "OpenVINO":
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = OVModelForCausalLM.from_pretrained(hf_model_id)
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)
else:
    precision = input("The model is not already in OpenVINO IR format. please select the precision of new compressed model (int4 or int8 or no_compression) and press enter ")
    if precision not in ["int4", "int8", "no_compression"]:
        logging.info("The precision must be int4 or int8 or no_compression")
        sys.exit()

    if precision == "int8":
        int8_model_dir = Path(local_model_path) / "int8"
        convert_to_int8()
    elif precision == "int4":
        int4_mode = input("Select the mode of int4 conversion (SYM or ASYM) and press enter")
        if int4_mode not in ["SYM", "ASYM"]:
            logging.info("The mode must be SYM or ASYM")
            sys.exit()
        group_size = input("Select the group size and press enter")
        if not group_size.isdigit() or int(group_size) < 1:
            logging.info("The group size must be an integer or greater than 0")
            sys.exit()
        group_size = int(group_size)

        ratio = input("Select the ratio (float between 0 and 1) and press enter: ").strip()
        try:
            ratio = float(ratio)
            if not (0 < ratio <= 1):
                raise ValueError
        except ValueError:
            print("The ratio must be a float between 0 and 1")
            sys.exit()
        int4_model_dir = Path(local_model_path) / "int4"
        convert_to_int4(int4_mode, group_size, ratio)

    elif precision == "no_compression":
        export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past".format(hf_model_id)
        export_command_base += " --trust-remote-code"
        export_command = export_command_base + " " + str(local_model_path)
        print(export_command)
        subprocess.run(export_command)


    

#TODO: re-download the model using trust_remote_code = True in from_pretrained function as suggested in HF documentation
#TODO: Aggiustare lo script migliorarlo (provare a tolgiere il comando optium-cli e usare le librerie di OpenVINO)