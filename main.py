import pandas as pd
import torch
from data_preprocessing import read_data
from model_making import  launch_model_training, make_predictions
from visualisation import make_loss_graph
from store_metrics import store_loss

if __name__ == '__main__':
    torch.set_default_device('cuda')  # Use GPU by default

    # Read in all datasets:
    train_x_set = pd.read_csv("data/arguments-training.tsv", sep="\t")
    val_x_set = pd.read_csv("data/arguments-validation.tsv", sep="\t")
    test_x_set = pd.read_csv("data/arguments-test.tsv", sep="\t")
    train_y_set = pd.read_csv("data/labels-training.tsv", sep="\t")
    val_y_set = pd.read_csv("data/labels-validation.tsv", sep="\t")
    test_y_set = pd.read_csv("data/labels-test.tsv", sep="\t")

    # Convert data to loaded tensors:
    loaded_torch_train_data = read_data(train_x_set, train_y_set)
    loaded_torch_val_data = read_data(val_x_set, val_y_set)
    loaded_torch_test_data = read_data(test_x_set, test_y_set)

    # Feed tensors into DeBERTa and perform training:
    # detailed_train_loss_list, train_loss_list, val_loss_list = launch_model_training(loaded_torch_train_data, loaded_torch_val_data)

    # Perform testing:
    test_f1, _ = make_predictions(loaded_data=loaded_torch_test_data, mode='test', model=None)
    print(f"Results on testset: {test_f1}")

    # Save data on the losses
    # store_loss([train_loss_list, val_loss_list], "train_val")
    # store_loss(detailed_train_loss_list, "detailed_train")

    # Print graphs on the loss:
    # make_loss_graph(detailed_train_loss_list, "Training", detailed = True)
    # make_loss_graph([train_loss_list, val_loss_list], "Training and validation", detailed = False)
    # make_loss_graph(train_loss_list, "Testing", detailed = False)
