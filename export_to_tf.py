""" 
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ File for transforming a Pytorch model (.pt or .pth) to tensorflowjs        │
  │ Step :                                                                     │
  │     1. Load your model                                                     │
  │     2. Export Pytorch Model to ONNX                                        │
  │     3. Export ONNX model to Tensorflow                                     │
  │     4. Exporte Tensorflow to tensorflowjs (commande)                       │
  └────────────────────────────────────────────────────────────────────────────┘
 """

import onnx
from onnx_tf.backend import prepare

ONNX_PATH = "./model/model.onnx"
TF_PATH = "./model/model_tf"

if __name__=="__main__":
    onnx_model = onnx.load(ONNX_PATH)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(TF_PATH)
