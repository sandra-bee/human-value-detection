import os

# Save content to a file:
def save_into_file(path, filename, file_content, mode='w'):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), mode) as f:
        f.write(file_content)

# Store the losses in separate files:
def store_list(list_of_vals, type_of_list, filename):
    save_into_file('result_metrics/', filename + f'_{type_of_list}', str(list_of_vals))

