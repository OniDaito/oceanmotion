# Explanation

OceanMotion is a collection of neural network models (mostly U-Net) and supporting utilities. The goal is to create a model that can reliably spot moving objects in sonar, specfically seals. We used the [Tritech Gemini](https://www.tritech.co.uk/products/gemini-720ik) sonars specifically when building this system.

It is written in the [Python](https://python.org) language and uses a number of supporting libraries, chiefly [PyTorch](https://pytorch.org/). The final model is designed to work with either PyTorch or the [ONNX Runtime](https://onnxruntime.ai/).

OceanMotion can reduce the time taken to spot these moving objects considerably - reducing the number of false positives.

In order to use OceanMotion you can download a pre-trained model, or train one yourself from scratch.

The code is [OpenSource](https://opensource.org/) and available on [GitHub](https://github.com/onidaito/oceanmotion).


## OceanMotion's pipeline

When ocean motion is processing a Group from the database, it goes through the following major steps:

1. Find all the .fits.lz4 files for this group and a particular sonar.
2. Decompress them.
3. Crop them to 512 x 1632.
4. Resize to 256 x 816.
5. Normalise the values to be in the range 0 to 1.
6. Build a queue of 16 images (or whatever the window size is set to).
7. Expand to dimensions (1, 1, 16, 816, 256) (B,C,D,H,W).
8. Make a prediction on this queue.
9. Pass the pixels of prediction through a sigmoid.
10. If the pixel >= confidence set to 1, else set to 0.
11. Pop the first element off the queue.
12. Push back a new element onto the queue.
13. Predict again.
14. Repeat until there are no more elements.

The result is a list of masks - images the same size as input but 0 or 1 - one for each input frame minus the initial 15 used to build the window. A 0 marks out the background, 1 the moving object.