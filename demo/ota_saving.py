# Prints the layers shared by releases v1.0.0 and v1.0.1 and the OTA bandwidth saving for 85 trucks.
import json
import subprocess
from pathlib import Path

TRUCKS = 85
RUPEES_PER_MB = 0.10


# Returns the podman inspect record of one image tag.
def inspect(tag):
    result = subprocess.run(["podman", "image", "inspect", tag], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)[0]


old = inspect("logibridge-inference:v1.0.0")
new = inspect("logibridge-inference:v1.0.1")
old_layers = old["RootFS"]["Layers"]
new_layers = new["RootFS"]["Layers"]
shared = 0
for layer in new_layers:
    if layer in old_layers:
        shared = shared + 1

image_mb = new["Size"] / 1e6
model_mb = Path("training/models/m3_pruned_int8.tflite").stat().st_size / 1e6
full_mb = image_mb * TRUCKS
layer_mb = model_mb * TRUCKS

print(f"Layers shared          : {shared} of {len(new_layers)} (only {len(new_layers) - shared} sent over the air)")
print(f"Full image per truck   : {image_mb:10.2f} MB")
print(f"Model layer per truck  : {model_mb:10.4f} MB")
print(f"85 trucks, full image  : {full_mb:10.1f} MB = Rs {full_mb * RUPEES_PER_MB:,.2f}")
print(f"85 trucks, model layer : {layer_mb:10.3f} MB = Rs {layer_mb * RUPEES_PER_MB:,.4f}")
print(f"Saving per update      : {(1 - layer_mb / full_mb) * 100:.4f}%")
print(f"Brief's 280 KB model   : {0.28 * TRUCKS:.1f} MB = Rs {0.28 * TRUCKS * RUPEES_PER_MB:.2f} per update")
