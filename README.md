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

The batch size during model training is set at 32, as this is the max size that didn't run out of memory on the test computer.

## TODO:
* Put data after augmentation into a dataframe
* Visualise results with confusion matrix(?)
* Tune hyperparameters: lr, eps, patience

## References

<a id="1">[1]</a>  https://touche.webis.de/semeval23/touche23-web/index.html
