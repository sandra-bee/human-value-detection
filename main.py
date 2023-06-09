import ast
import pandas as pd
import torch
from data_preprocessing import read_data
from model_making import launch_model_training, make_predictions
from visualisation import make_loss_graph
from store_metrics import store_list
import sys

MAKE_PLOTS = True
GRID_SEARCH = False
USE_DATA_AUG = False


# Launch training on specified hyperparameter values:
def init_training(lr, patience):
    detailed_train_loss, train_loss, val_loss, val_f1 = \
        launch_model_training(loaded_torch_train_data, loaded_torch_val_data,
                              learning_rate=lr, patience=patience)

    # Save data on the losses
    store_list([train_loss, val_loss], "loss",
               f"train_val_lr{lr}_ptn{patience}")
    store_list(detailed_train_loss, "loss",
               f"detailed_train_lr{lr}_ptn{patience}")

    # Save the f1 scores of the validation set
    store_list(val_f1, "f1",
               f"val_lr{lr}_ptn{patience}")

    return detailed_train_loss, train_loss, val_loss, val_f1


# Grid search across the learning rates and patience values:
def init_grid_search(hyperparams_dict):
    for i in range(len(hyperparams['lr'])):
        for j in range(len(hyperparams['patience'])):
            detailed_train_loss, train_loss, val_loss, val_f1 = \
                launch_model_training(loaded_torch_train_data, loaded_torch_val_data,
                                      learning_rate=hyperparams_dict['lr'][i], patience=hyperparams_dict['patience'][j])

            # Save data on the losses
            store_list([train_loss, val_loss], "loss",
                       f"train_val_lr{hyperparams_dict['lr'][i]}_ptn{hyperparams_dict['patience'][j]}")
            store_list(detailed_train_loss, "loss",
                       f"detailed_train_lr{hyperparams_dict['lr'][i]}_ptn{hyperparams_dict['patience'][j]}")

            # Save the f1 scores of the validation set
            store_list(val_f1, "f1",
                       f"val_lr{hyperparams_dict['lr'][i]}_ptn{hyperparams_dict['patience'][j]}")

    return detailed_train_loss, train_loss, val_loss, val_f1


if __name__ == '__main__':
    # This script is run by specifying 'python3 main.py <train|test|extra_test>:
    if len(sys.argv) < 1:
        print("Please run this script as: python3 main.py <train|test|extra_test>")
    else:
        run_mode = sys.argv[1]

    torch.set_default_device('cuda')  # Use GPU by default

    # Read in all datasets:
    if USE_DATA_AUG:
        train_x_set = pd.read_csv("data/augmented-arguments-training.tsv", sep="\t")
        train_y_set = pd.read_csv("data/augmented-labels-training.tsv", sep="\t")
    else:
        train_x_set = pd.read_csv("data/arguments-training.tsv", sep="\t")
        train_y_set = pd.read_csv("data/labels-training.tsv", sep="\t")

    val_y_set = pd.read_csv("data/labels-validation.tsv", sep="\t")
    val_x_set = pd.read_csv("data/arguments-validation.tsv", sep="\t")
    test_x_set = pd.read_csv("data/arguments-test.tsv", sep="\t")
    test_y_set = pd.read_csv("data/labels-test.tsv", sep="\t")

    test_x_nahj_set = pd.read_csv("data/arguments-test-nahjalbalagha.tsv", sep="\t")
    test_y_nahj_set = pd.read_csv("data/labels-test-nahjalbalagha.tsv", sep="\t")
    test_x_zhihu = pd.read_csv("data/arguments-validation-zhihu.tsv", sep="\t")
    test_y_zhihu = pd.read_csv("data/labels-validation-zhihu.tsv", sep="\t")
    test_x_nyt = pd.read_csv("data/arguments-test-nyt.tsv", sep="\t")
    test_y_nyt = pd.read_csv("data/labels-test-nyt.tsv", sep="\t")

    # Convert data to loaded tensors:
    loaded_torch_train_data = read_data(train_x_set, train_y_set)
    loaded_torch_val_data = read_data(val_x_set, val_y_set)
    loaded_torch_test_data = read_data(test_x_set, test_y_set)
    loaded_torch_nahj_data = read_data(test_x_nahj_set, test_y_nahj_set)
    loaded_torch_zhihu_data = read_data(test_x_zhihu, test_y_zhihu)
    loaded_torch_nyt_data = read_data(test_x_nyt, test_y_nyt)

    if run_mode == 'train':
        if GRID_SEARCH:
            # Feed tensors into DeBERTa and perform training through grid-search:
            hyperparams = {
                'lr': [5e-5, 5e-6, 5e-7],
                'patience': [11]
            }
            detailed_train_loss_list, train_loss_list, val_loss_list, _ = init_grid_search(hyperparams)

        else:
            # Feed tensors into DeBERTa and perform training on optimal hyperparameter values:
            detailed_train_loss_list, train_loss_list, val_loss_list, _ = init_training(lr=5e-5, patience=12)
            # Print graphs on the loss:
            if MAKE_PLOTS:
                make_loss_graph(detailed_train_loss_list, "Training", detailed=True)
                make_loss_graph([train_loss_list, val_loss_list], "Training and validation", detailed=False)
                make_loss_graph(train_loss_list, "Testing", detailed=False)
    elif run_mode == 'test':
        # Make loss plots for the best model
        if MAKE_PLOTS:
            loss_file = open('result_metrics/best_model_train_val_lr5e-05_ptn11_loss', 'r')
            best_model_loss = ast.literal_eval(loss_file.read())
            make_loss_graph(best_model_loss, "Training and validation", detailed=False)

        test_f1, _ = make_predictions(loaded_data=loaded_torch_test_data, mode='test', model=None)
        print(f"Results on testset: {test_f1}")
    elif run_mode == 'extra_test':
        test_nahj_f1, _ = make_predictions(loaded_data=loaded_torch_nahj_data, mode='test', model=None)
        print(f"Results on nahjalbalagha testset: {test_nahj_f1}")
        test_zhihu_f1, _ = make_predictions(loaded_data=loaded_torch_zhihu_data, mode='test', model=None)
        print(f"Results on zhihu testset: {test_zhihu_f1}")
        test_nyt_f1, _ = make_predictions(loaded_data=loaded_torch_nyt_data, mode='test', model=None)
        print(f"Results on nyt testset: {test_nyt_f1}")
    else:
        print("Please run this script as: python3 main.py <train|test|extra_test>")
