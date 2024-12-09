# PyTorch to TensorFlow.js Model Conversion Pipeline

This repository contains a pipeline to transform a PyTorch model (.pt) into a TensorFlow.js model. The process involves several steps:

1. Load the PyTorch model.
2. Export the PyTorch model to ONNX format.
3. Convert the ONNX model to TensorFlow format.
4. Convert the TensorFlow model to TensorFlow.js format.

Due to compatibility issues, we need two separate virtual environments for the conversion steps. Therefore, we have two different requirements files.

## Folder Structure

The `model` folder should contain the following files:

- `model.py`: Contains the network structure. The class should be named `Network`.
- `parameters.yaml`: Contains the parameters required for constructing the network (e.g., number of classes, hidden layers, etc.).
- `model.pt`: The PyTorch model file.

## Requirements

- `requirements_pt_to_tf.txt`: Contains the dependencies for converting PyTorch to TensorFlow.
- `requirements_tf_to_tfjs.txt`: Contains the dependencies for converting TensorFlow to TensorFlow.js.

## Steps

1. **Load the PyTorch model:**
   - Ensure your PyTorch model is saved in the `model` folder as `model.pt`.
   - The network structure should be defined in `model.py` with a class named `Network`.
   - The parameters for the network should be defined in `parameters.yaml`.

2. **Export PyTorch model to ONNX:**
   - Use the first virtual environment with dependencies from `requirements_pt_to_tf.txt`.
   - Run the script to export the PyTorch model to ONNX format.

3. **Convert ONNX model to TensorFlow:**
   - Use the first virtual environment with dependencies from `requirements_pt_to_tf.txt`.
   - Run the script to convert the ONNX model to TensorFlow format.

4. **Convert TensorFlow model to TensorFlow.js:**
   - Use the second virtual environment with dependencies from `requirements_tf_to_tfjs.txt`.
   - Run the script to convert the TensorFlow model to TensorFlow.js format.

## Example Bash Script

Here is an example bash script to automate the process:

```bash
#!/bin/bash

# Step 1: Export PyTorch model to ONNX
source venv_pt_to_tf/bin/activate
python export_to_onnx.py
deactivate

# Step 2: Convert ONNX model to TensorFlow
source venv_pt_to_tf/bin/activate
python convert_onnx_to_tf.py
deactivate

# Step 3: Convert TensorFlow model to TensorFlow.js
source venv_tf_to_tfjs/bin/activate
tensorflowjs_converter --input_format=tf_saved_model --output_node_names='output_node' --saved_model_tags=serve saved_model_dir web_model_dir
deactivate