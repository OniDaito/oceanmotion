# OceanMotion
This site contains the project documentation for the
`oceanmotion` project - a PyTorch-based neural network
for detecting moving objects in sonar images. 

## Table Of Contents

1. [Explanation](explanation.md) - The overall explanation of what sealhits does.
2. [Tutorials](tutorials.md) - Some basic tutorials on how to use the various scripts inside sealhits.
3. [How-To Guides](how-to-guides.md) - Specific how-tos on various things.
4. [Reference](reference.md) - An API reference.

## Requirements
You will need to install  **Python 3.11** - 3.12 is not supported yet, due to ONNX export (but this will change in the future).

We strongly recommend a virtual environment to run OceanMotion: **venv** or **miniconda** are good options. The requirements.txt lists all the python dependencies.

You will also need to install **[ffmpeg](https://ffmpeg.org/)** in order to view the results as predictions are presented both as NPZ files and webm videos.


The documentation follows the best practice for
project documentation as described by Daniele Procida
in the [Diátaxis documentation framework](https://diataxis.fr/).