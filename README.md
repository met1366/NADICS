**ADICS** is an open source framework for the purpose of Anyomaly Detection for Industrial Control Systems using classical machine learning and neural networks methods. The software adapts to Scikit-learn and TensorFlow, while taking general datasets as input. This flexibility lets you easily analyze training and inference on a single CPU or multiple GPUs in a desktop or server without rewriting code.

ADICS was originally developed by researchers and engineers working at the chair of IT Security at the Technical University of Munich for the purpose of conducting network intelligence research. The system is general enough to be applicable in a wide variety of other domains, as well.

## Installation
Just execute `python setup.py install`.

## Project structure

| **Folder**        | **Description**                                      |
| ------------- | ------------------------------------------------ |
| data          | Training and layout files should be stored here. |
| nadics       | The kernel code and configuration files.         |
| results       | In standard confguration the analysis files are written here. |

## Specifiying layout files
NADICS expects a layout file for each data set. For this purpose you can find `template.cfg` in `data/config`. E.g. the UNSW dataset in `data/example` requires the layout file `unsw.cfg` in `data/layouts`.

| **Attribute**          | **Description**     |
| ------------------ | --------------- |
| fullHeader     | All columns of the data set. |
| featureHeader  | Only the columns required for training. |
| binaryLabel    | One column containing the binary labels. |
| multiLabel    | One column containing the labels for multi classification. |
| stringFeatures | Only the columns containing strings in order to one-hot-encode. |

## Binary-classification with Random Forest

NADICS supports a variety of models do training and prediction single or multiple wise. Hereinafter we are showing how to execute the software with both methods and guide you through some of the outputs.

To start NADICS we choose binary-classification first entering `2` to classify between a *normal* and *malicious* attack. It follows one or more models to train and predict with as well as optional parameters. 

In order to enable you getting similar results we further parametrize our input with `-x 1.0 -y 1.0` to use the whole training and testing datasets. Adding `-w` will make sure any existing data that has been generated in a previous run will be overwritten.
```
python main.py 2 RandomForest -x 1.0 -y 1.0 -w
```
<img src="screenshots/one_model.png" width="70%" height="70%">

## Multi-classification with Random Forest

By using `-n` we are using multi-classification to distinguish between several attack scenarios.
```
python main.py n RandomForest -x 1.0 -y 1.0 -w
```
<img src="screenshots/multi_classification.png" width="60%" height="60%">

## Binary-classification with multiple models

Often we want to compare different models against each other using the same dataset. Therefore you can choose multiple different or
even the same models which are then run iteratevly. The end of the session will print a comparison of the chosen models with the best
and worst results highlighted.

Moreover you can use `-t` plus a number of seconds to determine when to cancel the execution with a single model, since e.g. the
Gaussian Process may take longer than expected to compute.
```
python main.py 2 RandomForest KNeighbors -x 1.0 -y 1.0 -w
```
<img src="screenshots/multiple_models.png" width="53%" height="53%">

## Analyzing feature importances

Datasets regarding network traffic are normally high dimensional. Hence we are interested in decreasing the number of dimensons to
reduce the computation time by still getting reliable results.

Therefore NADICS provides methods to output the importances of the features via console or to plot and write them to disk. 

By using `-i` we are printing all features with an importance greater or equal 1% onto the console.

```
python main.py 2 RandomForest -x 1.0 -y 1.0 -w -i
```
<img src="screenshots/importances_console.png" width="70%" height="70%">

By using `-p` NADICS stores plots formatted as `.svg` into `results/plots` folder and outputs when it is done.
```
python main.py 2 RandomForest -x 1.0 -y 1.0 -w -p
```
<img src="screenshots/importances_to_disk.png" width="70%" height="70%">

As you can see the plot also visualizes the standard deviation of each feature.
<img src="screenshots/importances_plot.png" width="83%" height="83%">



## Help

You can get more information about how to run NADICS by just calling `python main.py -h`.
