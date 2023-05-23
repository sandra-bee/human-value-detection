# About
This project attempts to solve the Human Value Detection 2023 task  [[1]](#1) using the DeBERTa v3 model.

# Usage
This project has been tested using Python 3.10 and is trained on a GPU. 

This project expects the following files in a `data` folder (not included in this repo, can be downloaded from [[1]](#1)):
* arguments-training.tsv
* arguments-validation.tsv
* arguments-test.tsv
* labels-training.tsv
* labels-validation.tsv
* labels-test.tsv

Install the requirements in `requirements.txt`, then run `main.py <train|test>` to launch. The parameter `train` will launch either grid search (if `GRID_SEARCH=True` in main) or will perform training with a set of optimal
hyperparameter values chosen. The parameter `test` will perform testing using the `best_model.pt` that has been saved in folder `models\` after training. You can set `MAKE_PLOTS=True` in main to visualise the training and validation loss obtained on the best model.

## Data Augmentation
To run data augmentation on the files in `\data`, run the file `DataAugmentation.py`. This will generate the following augmented training data files:
* augmented-arguments-training.tsv
* augmented-labels-training.tsv

You can then train the model using the augmented training data by setting the flag `USE_DATA_AUG=True` in main.

# Directory Structure
This project is structured as follows:
```bash
human-value-detection
│   ├── data
│   │   ├── arguments-training.tsv
│   │   └── ...
│   ├──models
│   │   ├── best_model.pt
│   │   └── ...
│   ├──plots
│   │   ├── Training and validation.png
│   │   └── ...
│   ├──result_metrics
│   │   ├── best_model_train_val_lr5e-05_ptn11_loss
│   │   └── ...
├── data_preprocessing.py
├── model_making.py
├── store_metrics.py
├── visualisation.py
├── LoadData.py
├── DataAugmentation.py
└── main.py
```

The `\data` folder contains the .tsv files containing the arguments and their respective labels (see 'Usage'). The data is loaded as a tensor for PyTorch in `data_preprocessing.py` and augmented in `LoadData.py` and `DataAugmentation.py`.

The `\models` folder contains the .pt PyTorch models that are generated after finetuning DeBERTa to solve this task. The models saved are the best ones (i.e. the model configuration where F1 score for the validation set was maximal.) The models are made in `model_making.py`.

The `\plots` folder contains the loss curves generated during training and validation. These plots are generated in `visualisation.py`.

The `\result_metrics` folder contains lists of F1 scores and loss values that are generated to make plots. E.g. the list of training losses per epoch are stored such that they can be plotted later. These lists are stored in `store_metrics.py`.

# Design Decisions
## Batch size
The batch size during model training is set at 16, as this is the max size that didn't run out of memory on the test computer.

## Data Augmentation

Training the models 3 times with learning rate 5e-5 and patience 10 and averaging the F1 scores, we obtained the following mean and standard deviations to 3 significant figures:

| With data augmentation | Without data augmentation |
|------------------------|---------------------------|
| 0.512 (±0.0128)        | 0.515 (±0.0161)           | 

Performing a paired t-test results in a p-value of: 0.910.
As this difference is insignificant, we design the training of the final model without data augmentation. 

## Optimal Hyperparameters
After running grid search while varying the hyperparameters `learning rate` and `patience`, we obtained the mean F1 scores (averaged over 3 runs) with standard deviations reported in the table below to 3 significant figures:

| Patience\Learning rate | __3__              | __4__            | __10__            | __11__           | __12__          |
|------------------------|--------------------|------------------|-------------------|------------------|-----------------|
| __5e-5__               | 0.496 (±0.0126)    | 0.509 (±0.0126)  | 0.515 (±0.0161)   | 0.522 (±0.0132)  | 0.502 (±0.0157) |
| __5e-6__               | 0.483   (±0.00676) | 0.491  (±0.0143) | 0.489   (±0.0113) | 0.502 (±0.00685) | x               |
| __5e-7__               | 0.418   (±0.0110)  | 0.417  (±0.0184) | 0.427  (±0.00700) | 0.450 (±0.0105)         | x               |

Learning rates larger than 5e-4 gave F1 scores 0 on the validation set so were not explored further as the learning process was too jumpy.

# References

<a id="1">[1]</a>  https://touche.webis.de/semeval23/touche23-web/index.html
