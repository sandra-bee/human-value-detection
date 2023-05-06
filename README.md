# About
This project attempts to solve the Human Value Detection 2023 task  [[1]](#1) using the DeBERTa v3 model.

## Usage
This project has been tested using Python 3.10 and is trained on a GPU. Run `main.py` to launch.
Note that this project expects the following files in a `data` folder (not included in this repo, can be downloaded from [[1]](#1)):
* arguments-training.tsv
* arguments-validation.tsv
* arguments-test.tsv
* labels-training.tsv
* labels-validation.tsv
* labels-test.tsv

## TODO:
* Data augmentation
* Write training/val losses to text file after generation so we can plot loss curves
* Visualise results with confusion matrix(?)
* Launch training on `deberta-v3-base` (instead of the `xsmall` version)
* Tune hyperparameters, add early stopping during training
* Improve the input. Currently, 2+ premises don't fit well with the template chosen
* Allow training on true max token length instead of capping early 

## References

<a id="1">[1]</a>  https://touche.webis.de/semeval23/touche23-web/index.html