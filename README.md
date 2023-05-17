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
hyperparameter values chosen. The parameter `test` will perform testing using the `best_model.pt` that is saved in folder `models\` after training.

# Directory Structure
This project is structured as follows:
```bash
human-value-detection
│   ├── data
│   │   ├── arguments-training.tsv
│   │   └── ...
│   ├──models
│   │   ├── best_model_lr0.005_ptn3.pt
│   │   └── ...
│   ├──plots
│   │   ├── Training.png
│   │   └── ...
│   ├──result_metrics
│   │   ├── train_val_lr0.005_ptn3_loss
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

## Optimal Hyperparameters
After running grid search while varying the hyperparameters `learning rate` and `patience`, we obtained the f1 scores (averaged over 3 runs) reported in the table below to 3 significant figures:

| Patience\Learning rate | __3__                  | __4__ | __10__   | 
|------------------------|------------------------|-------|----------|
| __5e-5__               | _training in progress_ |  _training in progress_     |    _training in progress_
| __5e-6__               | 0.484                  | 0.490 | 0.510 
| __5e-7__               |    _training in progress_                    |  _training in progress_     | _training in progress_

# TODO:
* Average results of 3 runs with grid search and add to optimal hyperparm table
* Run data augmentation to get augmented data back
* Visualise results with confusion matrix(?)


# References

<a id="1">[1]</a>  https://touche.webis.de/semeval23/touche23-web/index.html
