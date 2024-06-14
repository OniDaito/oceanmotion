# How-to guides

The following are a number of short guides on how perform some of the key processes in OceanMotion.

Each of the following guides assumes you have installed and loaded your Python virtual environment.

## Training a new model

To train a new model:

    python train.py -i ~/path/to/your/dataset -o ~/path/to/your/output -e 30 -b 4 -l 0.0001 --schedule -a -t UNetTRed


## Training a new model with sectored data

To train a new model but with sectors instead of pixel perfect tracks:

    python train.py -i ~/path/to/your/sectored/dataset -o ~/path/to/your/output -e 30 -b 4 -l 0.0001 --schedule -a -t Sector3D

## Recovering from a checkpoint

If, for some reason, your model crashed or failed to complete, you can recover from a checkpoint by adding **--checkpoint**.

    python train.py -i ~/path/to/your/dataset -o ~/path/to/your/output -e 30 -b 4 -l 0.0001 --schedule -a -t UNetTRed --checkpoint

## Running an existing model against a group in the database

    python run.py -o ~/tmp -f ~/path/to/the/fits/files -g your-group-huid-string -m ~/path/to/the/model.pt -t UNetTRed -p --img_width 256 --pred_length 16

You can ommit img_width adn pred_length if you are using the defaults (256 and 16 respectively).

## Convert to ONNX format

To convert a model to [ONNX](https://onnx.ai/) format, run the following command:

    python convert_to_onnx.py -i ~/sealz/datasets/2024_05_29 -o ~/sealz/runs/2024_05_29 -m ~/sealz/runs/2024_05_29/model.pt -t UNetTRed -d cpu

This will export a model that can be run on Linux with Python on the CPU. Replace **cpu** with **cuda** to get a cuda model. 

## Run an ONNX model in Python

The following will run the ONNX model against a group from the database:

    python run_onnx.py -g group-huid-we-want -t UNetTRed -o ~/path/to/output -m ~/path/to/model.onnx -f ~/path/to/fits


## Run an ONNX model with CUDA support in Python

Firstly, remove the cpu onnxruntime package:

    pip uninstall onnxruntime

Second, install the version of the ONNX Python runtime for CUDA12 (which my current system uses):

    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

Third, the onnxruntime-gpu needs the [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) from NVIDIA. This can be installed with pip:
    pip install tensorrt

It may also be that your Linux distro has the TensorRT software as a package inside it's package manager.

You can then run the same **run_onnx.py** command as before.

## Running an existing model against a time range and GLF files



## Running against a test set

It can be useful to get some insights into how well a model performs against the test set. To run a model against the entire set, execute the following command:

    python testset_test.py -m ~/path/to/model.pt -o ~/path/to/output -t UNetTRed -i ~/path/to/dataset

The output directory will contain numpy npz files of each prediction and a file *testset_results.csv* which contains a list of the files compared.

## Building and running the Java Wrapper

The Java wrapper is currently only working on Linux, as the program [lz4](https://github.com/lz4/lz4) is required to decompress the lz4 FITS files we use. Java's support for LZ4 is quite poor in comparison to Python and Rust, who's lz4 implementations were used to create the test data.

Firstly, make sure you have an ONNX version of the model you want to run. You can use the **convert_to_onnx.py** script mentioned above.

You'll need a Java JRE installed, along with the program Maven, as the example provided uses Maven to organise the build.

Secondly, you'll need to have the [sealhits_testdata]() repository downloaded as we'll be reading the FITS files included in that repos.

Navigate to **wrappers/java/oceanmotion** and run:

    mvn compile

Then, one can run the wrapper with maven as follows:

    mvn exec:java -Dexec.mainClass="oceanmotion.OceanMotion" -Dexec.args="/path/to/model.onnx /path/to/sealhits_testdata/fits"

The output of this program will be a video called **prediction.webm**.

## Building and running the C++ wrapper



## Visualising results


