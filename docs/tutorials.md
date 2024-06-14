# Tutorials

This tutorials page will take you through training and running a brand new OceanMotion model from scratch.

## Before we begin...
To train an OceanMotion model from scratch, you'll first need a dataset to train on.

This tutorial assumes you've ingested all the data you want from the PGDF, SQLite and GLF files using the program [SealHits](https://github.com/onidaito/sealhits). Secondly, it assumes you've generated a compatible dataset from the SealHits db and FITS files using the program [CrabSeal](https://github.com/onidaito/crabseal).

If you haven't done this, you'll need a dataset from somewhere else, such as the one available on [Zenodo](). This is definitely the easier option if you don't care about generating the dataset yourself.

OceanMotion is designed to run on Linux. OSX and Windows have not been tested. 

## Installing requirements.

Once you have a dataset in a particular directory, you'll need to setup a virtual environment of somekind for Python. I use the **venv** module as follows:

    python -m venv venv
    source ./venv/bin/activate

You can use [miniconda](https://docs.anaconda.com/free/miniconda/index.html) if you prefer, or any number of Python environment managers. Once you are setup, please run:

    pip install -r requirements.txt

This will install all the python requirements necessary to run OceanMotion.

OceanMotion makes considerable use of [ffmpeg](https://ffmpeg.org/) to generate it's video outputs. It's important to have this installed before you start. Under Linux, several package managers have ffmpeg available to install. Under OSX and Windows, binary installers are available.

## Training.

The script train.py contains the code necessary to build a model. To start training, run:

    python train.py -i ~/path/to/your/dataset -o ~/path/to/your/output -e 30 -b 4 -l 0.0001 --schedule -a -t UNetTRed

The parameters after train.py are:

* -i <str> - path to the input dataset.
* -o <str> - the output path for the model and log files.
* -e <uint> - the number of epochs to run for.
* -b <uint> - the batch size.
* -l <float> - the learning rate.
* --schedule - use the learning rate scheduler.
* -a - report to the website [Weights and Biases](https://wandb.ai)
* -t <string> - the model to use, in this case UNetTRed

The **-t** parameter refers to the models found in the file **model.py**. There are a number of models included - UNet3D, Sector3D, UNetTRed and UNetApricot. UnetTRed is the recommended model.

If you don't have (or don't want) a [Weights and Biases](https://wandb.ai) account, remove the **-a** switch. I find it very useful for tracking how well training is going, but it is not necessary.

There are a number of options you can pass to the training script. These are detailed at the end of the train.py file.

The time taken to train depends on:

1. The size of your dataset.
2. The number of epochs you want to run for.
3. The batch size.
4. The speed of your storage, GPU etc.
5. The size of your images.
6. How often you stop to run against the test set.

Typically, with an nVidia 4090, batch size 4, 256 x 816 pixel images, 20 epochs and a dataset of around 5000 items, training will take around 5 hours.

## Running the model.

Once you have finished training (or downloaded a pre-trained model), you should see a file called **model.pt** within the output directory you specified. This represents the saved model.

There are three major ways to run the model:

1. Against an existing group in our **SealHits** database.
2. Against a particular time period within a GLF file.
3. Continuously against a large number of GLF files.

The first option is the one we shall try as it allows us to compare our new model against an existing annotation.

    python run.py -o ~/tmp -f ~/path/to/the/fits/files -g your-group-huid-string -m ~/path/to/the/model.pt -t UNetTRed -p --img_width 256 --pred_length 16

The parameters have the following meanings:

* -o <str> - the output path.
* -f <str> - the path to the fits files.
* -g <str> - the Human ID (huid) of the group we are predicting against.
* -m <str> - the full path to the model.pt.
* -t <str> - the model type to load - *must be the same as the one in training and therefore in the model.pt*
* -p - generate a polar projection instead of raw rectangles.
* -r <int> - the sonar ID to 
* --img_width <int> - what width should the the raw FITS images be shrunk to. Should match that in the training.

Generally speaking, parameters such as **--pred_length** and **--img_width** should match these used in the training of the model. The defaults are usually sufficient - if you use the default parameters in training, you don't need to state them in the run.

If everything succeeds, there will be a number of files placed in the output directory:

* your-group-huid-string_sonarid_base.npz - the images as a compressed numpy npz array.
* your-group-huid-string_sonarid_ogmask.npz - the original mask as a compressed numpy npz array.
* your-group-huid-string_sonarid.gif - an animated gif of the prediction.
* your-group-huid-string_sonarid_pred.npz - the predicted mask as a compressed numpy npz array.
* your-group-huid-string_sonarid.webm - a video of the result.

The result video will show the sonar images, the original track (cyan) and the predicted track (red). When the tracks overlap, the colour is white.

Congratulations! You've built and tested the model. The how-to-guides describe how to run against a time range and perform longer term evaluations.