from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn import metrics
import numpy as np
import torch


def make_predictions(loaded_data, mode, model):
    # Use sigmoid with binary cross-entropy loss as we have a multi-label classification problem:
    loss_function = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mode == 'test':
        model = torch.load("best_model.pt")  # Load the best model obtained during testing
    model.eval()

    # Launch model testing:
    f1 = []
    loss = []
    with torch.no_grad():
        for batch in loaded_data:
            input_ids_batch, input_mask_batch, labels_batch = batch
            # Forward pass
            eval_output = model(input_ids=input_ids_batch, token_type_ids=None,
                                attention_mask=input_mask_batch, labels=labels_batch)

            label_ids = labels_batch.cpu().numpy()
            # Compute loss:
            loss_tensor = loss_function(eval_output.logits, labels_batch.to(device))
            loss.append(loss_tensor.item())
            # Get predictions by applying sigmoid + thresholding:
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(eval_output.logits.cpu())  # Running output through sigmoid to get probabilities
            preds = np.zeros(probs.shape)
            preds[np.where(probs >= 0.5)] = 1

            # Calculate f1 over all samples:
            curr_f1 = metrics.f1_score(label_ids, preds, average='samples')
            f1.append(curr_f1)

    mean_loss = sum(loss) / len(loss)
    mean_f1 = sum(f1) / len(f1)
    return mean_f1, mean_loss


def launch_model_training(loaded_train_data, loaded_val_data):

    loss_function = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 20

    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",
                                                          problem_type="multi_label_classification",
                                                          num_labels=num_labels)

    # Set learning rate very low, the model is already pretrained:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, eps=1e-08)
    max_epochs = 50
    best_f1 = 0
    train_loss_list = []
    detailed_train_loss_list = []
    val_loss_list = []
    patience = 4  # The number of times val loss can decrease from minimum until we run out of patience and stop early
    val_loss_prev_epoch = 100

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
            patience -= 1
            if patience == 0:
                break  # Stop early
        val_loss_prev_epoch = val_loss_curr_epoch

        # Save best model config:
        if val_f1_curr_epoch > best_f1:
            torch.save(model, 'best_model.pt')  # Save whole model
            best_f1 = val_f1_curr_epoch

        train_loss_list.append(train_loss_curr_epoch)
        val_loss_list.append(val_loss_curr_epoch)
        print(f'Training loss: {train_loss_curr_epoch}')
        print(f'Validation loss: {val_loss_curr_epoch}')
        print(f'Validation f1 score: {val_f1_curr_epoch}')

    return detailed_train_loss_list, train_loss_list, val_loss_list
