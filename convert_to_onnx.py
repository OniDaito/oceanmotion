#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

Author - Benjamin Blundell - bjb8@st-andrews.ac.uk

convert_to_onnx.py - convert a model.pt to an onnx binary.

"""

import os
import torch
import numpy as np
import importlib
from util.model import count_parameters, save_onnx


def main(args):
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    num_classes = 1
    ModelMoveType = getattr(importlib.import_module("model.model"), args.model_class)
    model = ModelMoveType(1, num_classes)

    assert os.path.exists(args.model_file)
    model.load_state_dict(torch.load(args.model_file, mmap=True), assign=True)
    model.to(device=args.device)
    model.eval()

    print("Number of model parameters:", count_parameters(model, no_grad=False))

    data_dir = os.path.join(args.data_path, "images/test")
    assert os.path.exists(data_dir)
    files_base = [
        f
        for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f)) and "base.npz" in f
    ]
    assert len(files_base) > 0
    base_npz = os.path.join(data_dir, files_base[0])
    source = np.load(base_npz)

    source = (
        torch.from_numpy(source)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .to(device=device)
    )
    source = source / 255.0  # Move to 0 to 1 range.

    save_onnx(model, source, args.out_path, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a model to an onnx format")
    parser.add_argument(
        "-i",
        "--data_path",
        default="./data",
        help="Path to the dataset (default: ./data)",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        default="./out",
        help="Path to save the output (default: ./out)",
    )
    parser.add_argument(
        "-m",
        "--model_file",
        default="./model.pt",
        help="Path to the model (default: ./model.pt)",
    )
    parser.add_argument(
        "-t",
        "--model_class",
        default="UNet3D",
        help="The model class to load (default: UNet3D)",
    )
    parser.add_argument(
        "-d", "--device", default="cuda", help="Which device to use (default: cuda)"
    )

    args = parser.parse_args()
    main(args)
