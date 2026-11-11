from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent

model_id = "BAAI/bge-reranker-v2-m3"
model_vendor, model_name = model_id.split("/")
path_to_model = ROOT / "reranker" / f"{model_name}_quantized"
print(path_to_model)

subprocess.run(
    ["optimum-cli", "export", "openvino", "--model", model_id, "--task", "text-classification",
     "--weight-format", "int8", str(path_to_model)],
    check=True,
)
