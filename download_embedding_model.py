import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

model_id = "BAAI/bge-small-en-v1.5"
path_to_model = ROOT / "models" / model_id
print(path_to_model)

subprocess.run(
    ["optimum-cli", "export", "openvino", "--model", model_id, "--task", "feature-extraction", str(path_to_model)],
    check=True,
)
