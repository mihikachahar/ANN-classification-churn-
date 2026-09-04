
import h5py
import json
import shutil

input_file = "model.h5"
output_file = "model_fixed.h5"

# Backup existing fixed model
try:
    shutil.copy(output_file, "model_fixed_backup.h5")
except:
    pass

with h5py.File(input_file, "r") as src:

    model_config = json.loads(src.attrs["model_config"])

    for layer in model_config["config"]["layers"]:

        config = layer["config"]

        # Fix InputLayer
        if layer["class_name"] == "InputLayer":

            # Original model input was 12 features
            config.pop("batch_shape", None)
            config.pop("batch_input_shape", None)
            config.pop("optional", None)

            config["shape"] = [12]

        # Fix Dense layers
        if layer["class_name"] == "Dense":

            # Remove unsupported config
            config.pop("quantization_config", None)

            # Fix GlorotUniform initializer
            kernel = config.get("kernel_initializer")

            if isinstance(kernel, dict):

                kernel_config = kernel.get("config", {})

                kernel_config.pop("input_axes", None)
                kernel_config.pop("output_axes", None)

    new_config = json.dumps(model_config)

    with h5py.File(output_file, "w") as dst:

        for key, value in src.attrs.items():

            if key == "model_config":
                dst.attrs[key] = new_config
            else:
                dst.attrs[key] = value

        src.copy("model_weights", dst)

print("SUCCESS!")
print("Created:", output_file)

