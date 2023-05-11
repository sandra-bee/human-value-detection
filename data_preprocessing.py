import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer


def retrieve_max_token_len(sentence_list):
    max_sentence_length = 0
    # Store the length of the longest sentence in terms of number of tokens:
    for sentence in sentence_list:
        if len(str(sentence).split()) > max_sentence_length:
            max_sentence_length = len(str(sentence).split())
    return max_sentence_length


def has_more_than_one_premise(text):
    num_full_stops = str(text).count('.')
    if num_full_stops > 2:
        return True
    return False


# Putting the data into the form of 1 concatenated sentence):
def read_data(x_set, y_set):
    concatenated_x_set = []
    for idx, row in x_set.iterrows():
        stance = "can" if x_set['Stance'][idx] == "in favor of" else "can not"

        if has_more_than_one_premise(x_set['Premise'][idx]):
            # Template for multiple premises:
            concatenated_x_set.append([f"Consider these ideas: '{x_set['Premise'][idx]}'. "
                                       f" From these, we {stance} conclude that {x_set['Conclusion'][idx]}."])
        else:
            # Template for one premise:
            concatenated_x_set.append([f"Based on the idea that {x_set['Premise'][idx]},"
                                             f" we {stance} conclude that {x_set['Conclusion'][idx]}."])

    max_sentence_length = retrieve_max_token_len(concatenated_x_set)

    # Encoding:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    # Set is_split_into_words=True because the concatenated_train_x_set is a list of strings
    encoding = tokenizer.batch_encode_plus(concatenated_x_set,
                                    add_special_tokens=True,
                                    max_length=max_sentence_length,
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

    return DataLoader(tensor_dataset, batch_size=16)