import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer


# Putting the data into the form of 1 concatenated sentence (1 input head):
def read_data(x_set, y_set):
    concatenated_x_set = []
    for idx, row in x_set.iterrows():
        stance = "can" if x_set['Stance'][idx]=="in favor of" else "can not"
        concatenated_x_set.append([f"based on the idea that {x_set['Premise'][idx]},"
                                         f" we {stance} conclude that {x_set['Conclusion'][idx]}"])

    # Encoding:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    # Set is_split_into_words=True because the concatenated_train_x_set is a list of strings
    encoding = tokenizer.batch_encode_plus(concatenated_x_set,
                                    add_special_tokens=True,
                                    max_length=128,
                                    return_token_type_ids=False,
                                    padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True,
                                     is_split_into_words=True)  # Max length to check

    # Prepare the labels:
    y_set.drop(columns="Argument ID", inplace=True)  # Drop arg id col
    # Use float for compatibility with loss metrics (BCE logits)
    targets = (torch.tensor(y_set.values.tolist(), dtype=torch.float))  # Convert the targets to a list

    input_id = torch.tensor(encoding['input_ids'])
    attention_mask = torch.tensor(encoding['attention_mask'])

    # Put the x and y together:
    tensor_dataset = TensorDataset(input_id, attention_mask, targets)

    return DataLoader(tensor_dataset, batch_size=32)


# Put the data in the form of 3 disconnected sentences:
# (This is not working well yet, need to see why)
def read_data_as_three_sep_sentences(x_set, y_set):
    concatenated_x_set = []
    for idx, row in x_set.iterrows():
        concatenated_x_set.append([f"{x_set['Conclusion'][idx]} [SEP] {x_set['Stance'][idx]} [SEP] {x_set['Premise'][idx]}"])
    # Flatten list:
    flat_conc_list = [item for sublist in concatenated_x_set for item in sublist]
    max_token_len_conc = 0
    for item in flat_conc_list:
        if len(item) > max_token_len_conc:
            max_token_len_conc = len(item)

    # Encoding:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
    # Set is_split_into_words=True because the concatenated_train_x_set is a list of strings
    # Batch_encode_plus will be able to interpret multiple sentences
    encoding = tokenizer.batch_encode_plus(concatenated_x_set,
                                    add_special_tokens=True,  # Add [CLS], [SEP] tokens
                                    max_length=200,
                                    # max_length=max_token_len_conc,
                                    return_token_type_ids=False,
                                    padding='max_length',  # Pad up to max token length of sentences
                                    truncation=True,
                                    return_attention_mask=True,  # Attention masks to know where padding is
                                     is_split_into_words=True)  # Max length to check

    # Prepare the labels:
    y_set.drop(columns="Argument ID", inplace=True)  # Drop arg id col
    # Use float for compatibility with loss metrics (BCE logits)
    targets = (torch.tensor(y_set.values.tolist(), dtype=torch.float))  # Convert the targets to a list

    input_id = torch.tensor(encoding['input_ids'])
    attention_mask = torch.tensor(encoding['attention_mask'])

    # Put the x and y together:
    tensor_dataset = TensorDataset(input_id, attention_mask, targets)

    return DataLoader(tensor_dataset, batch_size=32)