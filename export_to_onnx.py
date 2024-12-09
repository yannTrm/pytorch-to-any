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


import torch 

import yaml


MODEL_WEIGHT_PATH = "./model/model.pt"
MODEL_NETWORK_PATH = "./model/model.py"
MODEL_PARAMETER_PATH = "./model/parameters.yaml"

ONNX_PATH = "./model/model.onnx"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from model.model import Network, dummy_input




if __name__=="__main__":
    with open(MODEL_PARAMETER_PATH, 'r') as file:
        params = yaml.safe_load(file)

    model = Network(**params)

    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, weights_only=False))
    model = model.to(device)

    dummy_input = dummy_input.to(device)
    torch.onnx.export(model, dummy_input, ONNX_PATH, export_params=True, opset_version=11, do_constant_folding=True, input_names=['input'], output_names=['output'])

