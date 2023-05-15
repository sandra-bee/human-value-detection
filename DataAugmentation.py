import nlpaug.augmenter.word as naw
import pandas as pd
import random
from summa import keywords
from nltk.corpus import wordnet

# example text augmentation using nlpaug package
# adapted from usage examples: https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb
def example(original_text):
    # Replaces random words with synonym
    print("Synonym Replacement")
    aug = naw.SynonymAug(aug_src='wordnet', aug_max=1)
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

    ## Augmentation for noising
    # Replaces random words with antonym
    print("Antonym Replacement")
    aug = naw.AntonymAug(aug_max=1)
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

    # Random word swap
    print("Random word swap")
    aug = naw.RandomWordAug(action="swap", aug_max=1)
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

    # Random word addition
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert", aug_max=1)
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

    # Random word deletion
    print("Random word deletion")
    aug = naw.RandomWordAug(aug_max=1)
    augmented_text = aug.augment(original_text)
    print("Original:")
    print(original_text)
    print("Augmented Text:")
    print(augmented_text)

def synonym_swap(original_text):
    # Replaces random words with synonym
    aug = naw.SynonymAug(aug_src='wordnet', aug_max=1)
    augmented_text = aug.augment(original_text)
    return augmented_text

# The four noising methods
def antonym_swap(original_text):
    # Replaces random words with antonym
    aug = naw.AntonymAug(aug_max=1)
    augmented_text = aug.augment(original_text)
    return augmented_text

def words_swap(original_text):
    # Random word swap
    aug = naw.RandomWordAug(action="swap", aug_max=1)
    augmented_text = aug.augment(original_text)
    return augmented_text

def word_addition(original_text):
    # Random word addition
    aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert", aug_max=1)
    augmented_text = aug.augment(original_text)
    return augmented_text

def word_deletion(original_text):
    # Random word deletion
    aug = naw.RandomWordAug(aug_max=1)
    augmented_text = aug.augment(original_text)
    return augmented_text

# A better synonym swap using keyword extraction
def better_synonym_swap(original_text):
    # uses textrank to find keyword
    TR_keywords = keywords.keywords(original_text, scores=True)
    # find synonyms of keyword
    if len(TR_keywords) > 1:
        synonyms = []
        for syn in wordnet.synsets(TR_keywords[0][0]):
            for lm in syn.lemmas():
                synonyms.append(lm.name())
        # pick synonym at random and replace keyword with synonym in original text
        if len(synonyms) < 0:
            synonym = random.choice(synonyms)
            synonym = synonym.replace('_', ' ')
            augmented_text = original_text.replace(TR_keywords[0][0], synonym)
        else:
            augmented_text = original_text
    else: # it doesn't always seem to find a keyword, so in that case choose random word
        augmented_text = synonym_swap(original_text)
    return augmented_text


# Adds one of the four sources of noise (addition, deletion, swap or antonym)
# to the premise at random
def add_noise_to_premise(premise):
    i = random.randint(1, 4)
    if i == 1:
        new_premise = antonym_swap(premise)
    elif i == 2:
        new_premise = words_swap(premise)
    elif i == 3:
        new_premise = word_addition(premise)
    elif i == 4:
        new_premise = word_deletion(premise)
    return new_premise


def save_data(dataset):
    dataset.to_csv('data/augmented-arguments-training.tsv', sep="\t", index=False)


def save_labels(dataset):
    dataset.to_csv('data/augmented-labels-training.tsv', sep="\t", index=False)


def create_augmented_dataset_tsv():
    training_df = pd.read_csv('data/arguments-training.tsv', sep='\t')
    augmented_training_df = pd.DataFrame.copy(training_df)
    labels_df = pd.read_csv('data/labels-training.tsv', sep='\t')
    augmented_labels_df = pd.DataFrame.copy(labels_df)
    for i in range(0, len(training_df)):
    #for i in range(0, 5):
        premise = training_df.loc[i]['Premise']
        noised_premise = add_noise_to_premise(premise)
        new_argument = pd.DataFrame({
            "Argument ID": training_df.loc[i]["Argument ID"],
            "Conclusion": training_df.loc[i]["Conclusion"],
            "Stance": training_df.loc[i]["Stance"],
            "Premise": noised_premise,
        })
        new_labels = pd.DataFrame({
            "Argument ID": labels_df.loc[i]["Argument ID"],
            "Self-direction: thought": labels_df.loc[i]["Self-direction: thought"],
            "Self-direction: action": labels_df.loc[i]["Self-direction: action"],
            "Stimulation": labels_df.loc[i]["Stimulation"],
            "Hedonism": labels_df.loc[i]["Hedonism"],
            "Achievement": labels_df.loc[i]["Achievement"],
            "Power: dominance": labels_df.loc[i]["Power: dominance"],
            "Power: resources": labels_df.loc[i]["Power: resources"],
            "Face": labels_df.loc[i]["Face"],
            "Security: personal": labels_df.loc[i]["Security: personal"],
            "Security: societal": labels_df.loc[i]["Security: societal"],
            "Tradition": labels_df.loc[i]["Tradition"],
            "Conformity: rules": labels_df.loc[i]["Conformity: rules"],
            "Conformity: interpersonal": labels_df.loc[i]["Conformity: interpersonal"],
            "Humility": labels_df.loc[i]["Humility"],
            "Benevolence: caring": labels_df.loc[i]["Benevolence: caring"],
            "Benevolence: dependability": labels_df.loc[i]["Benevolence: dependability"],
            "Universalism: concern": labels_df.loc[i]["Universalism: concern"],
            "Universalism: nature": labels_df.loc[i]["Universalism: nature"],
            "Universalism: tolerance": labels_df.loc[i]["Universalism: tolerance"],
            "Universalism: objectivity": labels_df.loc[i]["Universalism: objectivity"],
        }, index=[0])
        augmented_training_df = pd.concat([augmented_training_df, new_argument], ignore_index=True)
        augmented_labels_df = pd.concat([augmented_labels_df, new_labels], ignore_index=True)

        synonym_swapped_premise = better_synonym_swap(premise)
        new_argument = pd.DataFrame({
            "Argument ID": training_df.loc[i]["Argument ID"],
            "Conclusion": training_df.loc[i]["Conclusion"],
            "Stance": training_df.loc[i]["Stance"],
            "Premise": synonym_swapped_premise,
        }, index=[0])
        augmented_training_df = pd.concat([augmented_training_df, new_argument], ignore_index=True)
        augmented_labels_df = pd.concat([augmented_labels_df, new_labels], ignore_index=True)
    print(augmented_training_df)
    save_data(augmented_training_df)
    save_labels(augmented_labels_df)


original_text = "The quick brown fox jumps over the lazy dog"
# better_synonym_swap(original_text)

create_augmented_dataset_tsv()
