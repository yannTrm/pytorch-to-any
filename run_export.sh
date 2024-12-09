#!/bin/bash

# Function to display usage
usage() {
  echo "Usage: $0 -p <path_to_pt_to_tf_env> -t <path_to_tfjs_env>"
  exit 1
}

# Parse command line arguments
while getopts ":p:t:" opt; do
  case $opt in
    p) PT_TO_TF_ENV="$OPTARG"
    ;;
    t) TFJS_ENV="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        usage
    ;;
    :) echo "Option -$OPTARG requires an argument." >&2
       usage
    ;;
  esac
done

# Check if both parameters are provided
if [ -z "$PT_TO_TF_ENV" ] || [ -z "$TFJS_ENV" ]; then
  usage
fi

# Step 1: Execute the export_to_onnx.py script
echo "Executing export_to_onnx.py..."
python3 export_to_onnx.py

# Step 2: Activate the virtual environment for PyTorch to TensorFlow
echo "Activating virtual environment for PyTorch to TensorFlow at $PT_TO_TF_ENV..."
source "$PT_TO_TF_ENV/bin/activate"

# Step 4: Execute the export_to_tf.py script
echo "Executing export_to_tf.py..."
python3 export_to_tf.py

# Deactivate the virtual environment
deactivate

# Step 5: Activate the virtual environment for TensorFlow.js
echo "Activating virtual environment for TensorFlow.js at $TFJS_ENV..."
source "$TFJS_ENV/bin/activate"

# Step 6: Convert the TensorFlow model to TensorFlow.js
echo "Converting TensorFlow model to TensorFlow.js..."
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./model/model_tf ./model/model_tfs

# Deactivate the virtual environment
deactivate

echo "Done."