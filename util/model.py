#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

model.py - utilities dealing with the model.
"""
import importlib
import os
import torch


def count_parameters(model, no_grad=False):
    """
    Return the number of parameters in this model

    Args:
        model: the model we are looking at.
        no_grad: if true, return all parameters. If false, just return the trainable ones.
    """
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or no_grad))


def read_classes_file(class_to_csv_path: str):
    """ Given a path to this CSV file, load the classes we are 
    detecting.
    
    Args:
        class_to_csv_path: the path to the CSV file.
    """
  
    # We set the first 0 class to background and treat it as any other in this version
    id_to_class = {}
    id_to_class[0] = "background"

    assert(os.path.exists(class_to_csv_path))

    with open(class_to_csv_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            tokens = line.replace("\n", "").split(",")
            id_to_class[int(tokens[1])] = tokens[0]
            print(tokens[1], "id corresponds to", tokens[0])

    num_classes = len(id_to_class.keys())
    print("Number of classes:", num_classes)
   
    return (id_to_class, num_classes)


def save_onnx(model, data:torch.Tensor, out_path:str, device:str):
    """Save the model as an ONNX binary - model_moves.onnx.
    
    Args:
        model (): the model we are saving.
        data (torch.Tensor): An example training datum.
        out_path: path to the output model, not including the filename.
        device (str): the device we are running on.
    """
    datas = data.to(device)
    onnx_program = torch.onnx.dynamo_export(model, x=datas)
    onnx_program.save(os.path.join(out_path, "model.onnx"))
      

def load_model_pt(model_path: str, device:str, model_class="UNet3D"):
    """ Load a model from disk, and place in evaluation mode.
    
    Args:
        model_path (str): the path to the model we are loading, including the filename.
        device (str): the device we are running on.
        model_class (str): The class name of the model we are loading.
    """
    ModelType = getattr(importlib.import_module("model.model"), model_class)
    model = ModelType(1, 1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model