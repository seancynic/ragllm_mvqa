import pandas as pd
import jsonlines
import nltk
import json

# Define stopwords
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load the JSONL file
raw_file_path = 'llava_json_data/test-clean-answer.jsonl'
jsonl_file_path = 'llava_json_data/PC_avg_IVFPQ/test_clean_answer_top5_img.jsonl'
# Load the DataFrame
df = pd.read_csv('../SOURCE/PMC-VQA/test_clean.csv')

# Function to compare two strings
def compare_strings(text, answer):
    # Tokenize strings into words
    text_words = set(nltk.word_tokenize(text.lower()))
    answer_words = set(nltk.word_tokenize(answer.lower()))

    # Remove stopwords from answer words
    answer_words_without_stopwords = answer_words - stop_words

    # Check if all remaining words in answer appear in text
    if answer_words_without_stopwords.issubset(text_words):
        return True
    else:
        return False

# Function to check if answer is a substring of text
def is_substring(answer, text):
    return answer in text

raw_list = []
# Iterate over each row of the DataFrame
count = 0
# Open the JSONL file and iterate over each JSON object
with jsonlines.open(raw_file_path) as reader:

    for index, (row, json_object) in enumerate(zip(df.iterrows(), reader)):
        # Extract the "text" attribute from the JSON object
        text = json_object["text"].lower()

        # Extract the "Answer" from the DataFrame
        answer = df.loc[index, "Answer"].lower()

        # Check if the "Answer" is a substring of "text"
        if compare_strings(text, answer):
            # print(json_object["question_id"].lower())
            # print(df.loc[index, "Question"].lower().strip())
            raw_list.append(json_object["question_id"])
            count += 1

print(count / len(df))


jsonl_list = []
count = 0
with jsonlines.open(jsonl_file_path) as reader:

    for index, (row, json_object) in enumerate(zip(df.iterrows(), reader)):
        # Extract the "text" attribute from the JSON object
        text = json_object["text"].lower()

        # Extract the "Answer" from the DataFrame
        answer = df.loc[index, "Answer"].lower()

        # Check if the "Answer" is a substring of "text"
        if compare_strings(text, answer):
            # print(json_object["question_id"].lower())
            # print(df.loc[index, "Question"].lower().strip())
            jsonl_list.append(json_object["question_id"])
            count += 1

print(count / len(df))



set1 = set(raw_list)
set2 = set(jsonl_list)
intersection = set1.intersection(set2)
union = set1.union(set2)
overlap_rate = len(intersection) / len(set1)
print(overlap_rate)