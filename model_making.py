from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn import metrics
import numpy as np
import torch

def make_predictions(loaded_data, mode, model):
    # Use sigmoid with binary cross-entropy loss as we have a multi-label classification problem:
    loss_function = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mode == 'test':
        model = torch.load("models/best_model.pt", map_location=torch.device('cpu'))  # Load the best model obtained during training
    model.eval()

    # Launch model testing:
    f1 = []
    precision = []
    recall = []
    loss = []
    accuracy = []

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loaded_data:
            input_ids_batch, input_mask_batch, labels_batch = batch
            # Forward pass
            eval_output = model(input_ids=input_ids_batch, token_type_ids=None,
                                attention_mask=input_mask_batch, labels=labels_batch)

            label_ids = labels_batch.cpu().numpy()
            all_labels.extend(label_ids)
            # Compute loss:
            loss_tensor = loss_function(eval_output.logits, labels_batch.to(device))
            loss.append(loss_tensor.item())
            # Get predictions by applying sigmoid + thresholding:
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(eval_output.logits.cpu())  # Running output through sigmoid to get probabilities
            preds = np.zeros(probs.shape)
            preds[np.where(probs >= 0.5)] = 1

            all_preds.extend(preds)

            # Calculate f1 over all samples:
            curr_f1 = metrics.f1_score(label_ids, preds, average='samples')
            curr_precision = metrics.precision_score(label_ids, preds, average='samples')
            curr_recall = metrics.recall_score(label_ids, preds, average='samples')
            curr_accuracy = metrics.accuracy_score(label_ids, preds)
            f1.append(curr_f1)
            precision.append(curr_precision)
            recall.append(curr_recall)
            accuracy.append(curr_accuracy)
    mean_loss = sum(loss) / len(loss)
    mean_f1 = sum(f1) / len(f1)
    mean_recall = sum(recall) / len(recall)
    mean_precision = sum(precision) / len(precision)
    mean_accuracy = sum(accuracy) / len(accuracy)
    print("mean recall: " + str(mean_recall))
    print("mean precision: " + str(mean_precision))
    print("mean accuracy: " + str(mean_accuracy))

    print(metrics.classification_report(all_labels, all_preds))

    return mean_f1, mean_loss


def launch_model_training(loaded_train_data, loaded_val_data, learning_rate, patience):
    init_patience = patience

    loss_function = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 20

    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",
                                                          problem_type="multi_label_classification",
                                                          num_labels=num_labels)

    # Set learning rate very low, the model is already pretrained:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    max_epochs = 1000
    best_f1 = 0
    train_loss_list = []
    detailed_train_loss_list = []
    val_loss_list = []
    val_loss_prev_epoch = 100
    val_f1_list = []

    for epoch in range(max_epochs):

        # Launch model training:
        print(f"***EPOCH: {epoch}/{max_epochs}:***")
        model.train()
        training_loss = []

        for step, batch in enumerate(loaded_train_data):
            input_ids_batch, input_mask_batch, labels_batch = batch
            optimizer.zero_grad()
            # Forward pass:
            train_output = model(input_ids=input_ids_batch, token_type_ids=None,
                                 attention_mask=input_mask_batch, labels=labels_batch)
            # Backward pass:
            loss_tensor = loss_function(train_output.logits, labels_batch.to(device))
            loss_tensor.backward()
            # Update loss:
            optimizer.step()
            training_loss.append(loss_tensor.item())
            # Also save loss per step for plotting:
            detailed_train_loss_list.append(loss_tensor.item())

        train_loss_curr_epoch = sum(training_loss) / len(training_loss)

        # Launch model evaluation:
        model.eval()
        val_f1_curr_epoch, val_loss_curr_epoch = make_predictions(loaded_data=loaded_val_data, mode='validation', model=model)

        # Early stopping:
        if epoch > 0 and val_loss_curr_epoch > val_loss_prev_epoch:
            # Patience is the number of times val loss can be larger than in previous run
            if patience == 0:
                print(f"***Early stopping at: {epoch}/{max_epochs}***")
                break  # Stop early
            patience -= 1
        val_loss_prev_epoch = val_loss_curr_epoch

        # Save best model config:
        if val_f1_curr_epoch > best_f1:
            torch.save(model, f'models/best_model_lr{learning_rate}_ptn{init_patience}.pt')  # Save whole model
            best_f1 = val_f1_curr_epoch

        train_loss_list.append(train_loss_curr_epoch)
        val_loss_list.append(val_loss_curr_epoch)
        val_f1_list.append(val_f1_curr_epoch)
        print(f'Training loss: {train_loss_curr_epoch}')
        print(f'Validation loss: {val_loss_curr_epoch}')
        print(f'Validation f1 score: {val_f1_curr_epoch}')

    return detailed_train_loss_list, train_loss_list, val_loss_list, val_f1_list
