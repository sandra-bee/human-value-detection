import os

# Save content to a file:
def save_into_file(path, filename, file_content, mode='w'):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), mode) as f:
        f.write(file_content)

# Store the losses in separate files:
def store_loss(loss, filename):
    save_into_file('result_metrics/', filename + '_loss', str(loss))

