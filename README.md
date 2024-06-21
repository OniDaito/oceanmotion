# OceanMotion

A neural network that attempts to spot motion in sonar videos. It does so by employing a U-Net model to segment the areas of moving object from the background.

This version has support for the binary case (moving, not moving) and the multiclass case (background, moving object of type 1, moving object of type 2 etc).

## Usage

Firstly, a model must be trained on a particular dataset. These datasets are created by the program [CrabSeal](https://github.com/onidaito/crabseal).

In order to use the ONNX routines, *you'll need to install Python 3.11* as 3.12 isn't supported by PyTorch ONNX export yet.

Assumming you have run CrabSeal correctly, you should have a dataset of numpy npz files, split into train, test and validation sets. The first program to run is train.py. You'll need to setup a *python virtual environment* and install the dependencies. Conda and virtualenv are two such methods. Using virtualenv:

    python -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt

With your virtual environment setup, you can start to train.

### train.py

This program will train a new model on the dataset. To do this, execute a command similiar to this:

    python train.py -i /path/to/your/dataset -o /path/for/your/output -e 40 -b 2 -l 0.0001 -a

The following flags in that command are as follows:

* -e <num> - the number of epochs to run for.
* -b <num> - the batch size.
* -l <num> - the learning rate.
* -a - send reports to [Weights and Biases](https://wandb.ai/) - currently the only stats output.

The script should run and generate a model.pt file in the output directory. If it does not, due to a crash, a checkpoint file should exist. You can resume training by adding the *--checkpoint* flag. 

### run.py

This script runs the model against either a known group in the database, or a time period within a set of GLF files. Run performs the following actions that need user guidance:

* Reads either a GLF or set of FITS file to build the input volume.
* Crop the raw images down to a fixed height. This is typically 1632 pixels tall, but can be set by the user. This should match the original crop on the image the model was trained on.
* Optionally resize images. This should match the resize on the images that the model was trained on.
* There should be enough frames to fill the window. This window will have been set in training - usually 16 but can be 8 or 32.
* This volume is passed through the 

To run a model against an existing group held in a local database, execute the following:

    python run.py -m /path/to/model.pt -r 854 -g "become-high-member-family" -c 0.8 -p -o /path/to/output -f /path/to/fits/files --img_height 816 --crop_height 1632

Three things must be passed into the run.py that have no defaults - the path to the fits files, the path to the saved model and the huid for the group being run. A number of other parameters are also useful to set (but they have reasonable defaults):

* -c <num> - the confidence level in the case of the binary segmentation tast.
* -r <num> - the sonar we are using
* -p - generate polar distorted plots instead of sticking with rectangles.
* --pred_length <num> - how long was the window when this model was trained? Normally 16.

To run against a set of existing GLFs, a time range must be supplied as well as the path to the GLFS:

    python run.py -m /path/to/model.pt -r 854 -l /path/to/glfs -c 0.8 -p -o /path/to/output -f /path/to/fits/files --img_height 816 -a "2024-01-01 01:01:01.001" -b "2024-01-01 02:01:01.001"

The output will be a set of files including numpy npz and webm video.

### fast_eval.py

This script, so called because it was aiming to be fast, runs a model against a series of GLF files, saving the recorded *detections* in an SQLite file. This is useful when testing the model against long time frames - days, weeks and months. The format is similar to run:

    python fast_eval.py -m /path/to/model.pt -o /path/for/output -l /path/to/glfs -w 16 -x 256 -y 829 -r 854 -c 0.8 -a "2023-04-01 00:00:00" -b "2023-04-02 00:00:00"


### visualise.py

A web-based visualisation program is available that takes the output from fast_eval.py (the sqlite file) and plots the results on an interactive graph. It can be launched as follows:

    bokeh serve --show visualise.py --args -s ~/path/to/sqlite.sql -u <db username> -w <db password> -n <database host>

## Running in Docker, possibly on a cluster.

It is possible to run OceanMotion on a cluster using Docker. A dockerfile is included, based on the nvcr.io/nvidia/pytorch:23.10-py3 container. 

The script to train such a model will depend on your cluster setup, but it may look something like this:

    #!/bin/bash
    module load rootless-docker
    start_rootless_docker.sh --quiet
    docker run -v $HOME/oceanmotion:/oceanmotion -w /oceanmotion --gpus 1 --shm-size=1g -it --rm oceanmotion python fast_eval.py -m /path/to/trained/model.pt -o /path/to/output -g /path/to/glfs -w 16 -x 256 -y 829 -r 854 -c 0.8 -s "2023-04-01 00:00:00" -e "2023-04-02 00:00:00"

## Tests

The small suite of tests requires the [sealhits_testdata](https://github.com/OniDaito/sealhits_testdata) repository. Once you have this, export the following environment variable, then run pytest as follows:

    export SEALHITS_TESTDATA_DIR=/path/to/sealhits/test/data
    pytest
