#!/usr/bin/env python
"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

testset_test.py

Run over the entire test set from a particular run.
We generate a set of npzs of predictions.

Example usage:
    
    python testset_test.py -m ~/path/to/model/model.pt -o ~/path/to/output -t UNetTRed -i ~/path/to/dataset

"""

import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
from data.dataset_presplit import MoveDatasetPreSplit, SetType
from util.model import load_model_pt


def main(args):
    """ Run through the test set, generate results and create a pretty picture."""
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    # Now load the model
    assert(os.path.exists(args.model_path,))
    model = load_model_pt(args.model_path, device, args.model_class)

    # Load our dataset
    dataset_test = MoveDatasetPreSplit(args.data_path, settype=SetType.TEST, num_classes=1)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        shuffle=False,
    )
 
    with open(os.path.join(args.out_path, "testset_results.csv"), "w") as f:
        f.write("source,target,pred\n")

        for idx, (source, target) in enumerate(tqdm(test_loader, desc="Running test loss")):
            source = source.to(device)
            target = target.to(device)
            target_binary = torch.where(target > 0, 1, 0).to(dtype=torch.uint8)
            pred = model(source)
            pred = torch.where(F.sigmoid(pred) > args.confidence, 1, 0)
            pred = pred.squeeze().cpu().detach().long().numpy().astype(np.uint8)

            source = source.squeeze().cpu().detach().numpy()
            source = source * 255
            source = source.astype(np.uint8)
            target = target.squeeze().cpu().detach().numpy().astype(np.uint8)

            source_name, target_name = dataset_test.files[idx]
            pred_name = source_name.split("_half")[0] + "_half_pred.npz"

            np.savez_compressed(os.path.join(args.out_path, source_name), x=source)
            np.savez_compressed(os.path.join(args.out_path, target_name), x=target_binary.cpu())
            np.savez_compressed(os.path.join(args.out_path, pred_name), x=pred)

            f.write(source_name + "," + target_name + "," + pred_name + "\n")
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="oceanmotion - settest",
        description="Run a model over a test set file and get losses and a video back.",
        epilog="SMRU St Andrews",
    )
    parser.add_argument(
        "-m", "--model_path", default=".", help="The path to the directory of the run"
    )
    parser.add_argument(
        "-o", "--out_path", default=".", help="The path for the output."
    )
    parser.add_argument(
        "--sector",
        action="store_true",
        default=False,
        help="Rather than predict full paths, predict sectors(default: false)",
    )
    parser.add_argument(
        "-t",
        "--model_class",
        default="UNet3D",
        help="The model class to load (default: UNet3D)",
    )
    parser.add_argument(
        "-s",
        "--set_length",
        type=int,
        default=16,
        help="The length of the strips (default: 16)",
    )
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.7,
        help="What confidence level do we want (default: 0.7)",
    )
    parser.add_argument("-i", "--data_path", default="./data")

    args = parser.parse_args()

    main(args)
