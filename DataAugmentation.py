import LoadData
from datasets import Dataset, concatenate_datasets
import nlpaug.augmenter.word as naw

# example text augmentation using nlpaug package
# adapted from usage examples: https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb
def example(original_text):
    # Replaces random words with synonym
    print("Synonym Replacement")
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

    ## Augmentation for noising
    # Replaces random words with antonym
    print("Antonym Replacement")
    aug = naw.AntonymAug()
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

    # Random word swap
    print("Random word swap")
    aug = naw.RandomWordAug(action="swap")
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

    # Random word addition
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert")
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

    # Random word deletion
    print("Random word deletion")
    aug = naw.RandomWordAug()
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

def synonym_swap(original_text):
    # Replaces random words with synonym
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(original_text)
    return augmented_text


def antonym_swap(original_text):
    # Replaces random words with antonym
    aug = naw.AntonymAug()
    augmented_text = aug.augment(original_text)
    return augmented_text

def words_swap(original_text):
    # Random word swap
    aug = naw.RandomWordAug(action="swap")
    augmented_text = aug.augment(original_text)
    return augmented_text

def word_addition(original_text):
    # Random word addition
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert")
    augmented_text = aug.augment(original_text)
    return augmented_text

def word_deletion(original_text):
    # Random word deletion
    aug = naw.RandomWordAug()
    augmented_text = aug.augment(original_text)
    return augmented_text

def gen():
    dataset_size = int(LoadData.training_dataset.num_rows) -1
    for i in range(0, dataset_size):
        premise = LoadData.training_dataset[i]["Premise"]
        # currently runs once (1 new argument for each argument) can be increased depending on training limits
        for _ in range(1):
            augmented_premise = synonym_swap(premise)
            augmented_premise = " ".join(augmented_premise)
            yield {"Argument ID": LoadData.training_dataset[i]["Argument ID"],
                   "Conclusion": LoadData.training_dataset[i]["Conclusion"],
                   "Stance": LoadData.training_dataset[i]["Stance"],
                   "Premise": augmented_premise,
                   "Labels": LoadData.training_dataset[i]["Labels"]}


# currently takes very long to run
def gen_noise():
    dataset_size = int(LoadData.training_dataset.num_rows) -1
    for i in range(0, dataset_size):
        premise = LoadData.training_dataset[i]["Premise"]
        augmented_premise = antonym_swap(premise)
        augmented_premise = " ".join(augmented_premise)
        yield {"Argument ID": LoadData.training_dataset[i]["Argument ID"],
               "Conclusion": LoadData.training_dataset[i]["Conclusion"],
               "Stance": LoadData.training_dataset[i]["Stance"],
               "Premise": augmented_premise,
               "Labels": LoadData.training_dataset[i]["Labels"]}
        augmented_premise = words_swap(premise)
        augmented_premise = " ".join(augmented_premise)
        yield {"Argument ID": LoadData.training_dataset[i]["Argument ID"],
               "Conclusion": LoadData.training_dataset[i]["Conclusion"],
               "Stance": LoadData.training_dataset[i]["Stance"],
               "Premise": augmented_premise,
               "Labels": LoadData.training_dataset[i]["Labels"]}
        augmented_premise = word_deletion(premise)
        augmented_premise = " ".join(augmented_premise)
        yield {"Argument ID": LoadData.training_dataset[i]["Argument ID"],
               "Conclusion": LoadData.training_dataset[i]["Conclusion"],
               "Stance": LoadData.training_dataset[i]["Stance"],
               "Premise": augmented_premise,
               "Labels": LoadData.training_dataset[i]["Labels"]}
        augmented_premise = word_addition(premise)
        augmented_premise = " ".join(augmented_premise)
        yield {"Argument ID": LoadData.training_dataset[i]["Argument ID"],
               "Conclusion": LoadData.training_dataset[i]["Conclusion"],
               "Stance": LoadData.training_dataset[i]["Stance"],
               "Premise": augmented_premise,
               "Labels": LoadData.training_dataset[i]["Labels"]}



def create_augmented_dataset():
    synonym_augmented_data = Dataset.from_generator(gen)
    print(synonym_augmented_data)
    print(synonym_augmented_data[0:5])
    noised_data = Dataset.from_generator(gen_noise)
    print(noised_data)
    print(noised_data[0:5])
    complete_data = concatenate_datasets([LoadData.training_dataset, synonym_augmented_data, noised_data])
    return complete_data


#original_text = "The quick brown fox jumps over the lazy dog"
#example(original_text)

augmented_dataset = create_augmented_dataset()
