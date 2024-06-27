import json
import torch
from torch.utils.data import DataLoader
import os
import string
from finetuning_dataloader import CustomDataset

json_data = "train_features_lrec_camera.json"
batch_size = 2
shuffle = False
text_pad_length=500
img_pad_length=36
audio_pad_length=63

dataset = CustomDataset(json_data, text_pad_length, img_pad_length, audio_pad_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

data_entry = next(iter(dataloader))
for key, item in data_entry.items():
    print(key)
    print(item)
    print(item.shape)
    print()

# TRAINING SET INFO
# 2890 EXAMPLES, 31 NO WORDS

# Define the base directory as the directory where the script is located
# base_dir = os.path.dirname(os.path.abspath(__file__))

# # Define the processed data directory relative to the base directory
# processed_data_dir = os.path.join(base_dir, "processed_data")

# # Define the training features file path
# training_features = os.path.join(processed_data_dir, 'train_features_lrec_camera.json')
# val_features = os.path.join(processed_data_dir, 'val_features_lrec_camera.json')
# test_features = os.path.join(processed_data_dir, 'test_features_lrec_camera.json')


# label_to_idx = {0: 0, 1: 1}
# idx_to_label = { 0: 0, 1: 1}

# #label_to_idx = {'R': 0, 'PG-13': 1,  'PG':2}
# #idx_to_label = { 0: 'R', 1: 'PG-13', 2: 'PG'}

# genre_to_idx = {'Sci-Fi':0, 'Crime': 1, 'Romance': 2, 'Animation': 3, 'Music': 4, 'Adult': 5, 'Comedy': 6, 'War': 7, 'Horror': 8, 'Film-Noir': 9, 'Adventure': 10, 'News': 11, 'Thriller': 12, 'Western': 13, 'Mystery': 14, 'Short': 15, 'Drama': 16, 'Action': 17, 'Documentary': 18, 'History': 19, 'Family': 20, 'Fantasy': 21, 'Sport': 22, 'Biography': 23, 'Musical':4, 'Talk-Show':24}
# idx_to_genre = {0:'Sci-Fi', 1:'Crime', 2:'Romance', 3:'Animation', 4:'Music,Musical', 5:'Adult', 6:'Comedy', 7:'War', 8:'Horror', 9:'Film-Noir', 10:'Adventure', 11:'News', 12:'Thriller', 13:'Western', 14:'Mystery', 15:'Short', 16:'Drama', 17:'Action', 18:'Documentary', 19:'History', 20:'Family', 21:'Fantasy', 22:'Sport', 23:'Biography', 24:'Musical',25:'Talk-Show'}

# idx_to_emotion = {0: 'positive', 1: 'sadness', 2: 'joy', 3: 'trust', 4: 'fear', 5: 'negative', 6: 'surprise', 7: 'anger', 8: 'anticipation', 9: 'disgust'}
# emotion_to_idx = {'positive': 0, 'sadness': 1, 'joy': 2, 'trust': 3, 'fear': 4, 'negative': 5, 'surprise': 6, 'anger': 7, 'anticipation': 8, 'disgust': 9}

# fold = "1"

# # the punctuation code here isn't really working but whatever
# def list_to_text(words):
#     # Define punctuation characters
#     punctuation = set(string.punctuation)

#     result = ""
#     for i, word in enumerate(words):
#         if i > 0 and word not in punctuation:
#             # Add a space before the word if it's not punctuation
#             result += " "
#         result += word

#     return result

# batch_size = 16

# device = torch.device('cpu')
# if torch.cuda.is_available():
#     device = torch.device('cuda')

# features_dict_train = json.load(open(training_features))
# features_dict_val = json.load(open(val_features))
# features_dict_test = json.load(open(test_features))

# train_dict_iter = iter(features_dict_train.items())
# # print(len(features_dict_train))
# key, value = next(train_dict_iter)
# # print(key)

# print(value.keys())
# # print(value)

# no_words_cnt = 0
# for index, i in enumerate(features_dict_train):
#     # verify that i and mid in nlp_comic_binary are the same - they are
#     # if i != features_dict_train[i]["IMDBid"]:
#     #     print(f"MISMATCH AT {index}")

#     # investigate the weird if condition
#     # if i == "laqIl3LniQE.02":
#     #     print(features_dict_train[i])

#     if len(features_dict_train[i]["words"]) == 0:
#         no_words_cnt += 1
# print(no_words_cnt) # 31 


# print(value["words"])
# print()
# print(value["cap_words"])

# One entry Looks as follows
# Key: A strange string representing the id (used to get video and audio vectors later)
# Value is a dict
# id: some number (not sure the meaning or where it's used)
# indexes: presumably the tokenized text
# IMBid: same as the key
# label: objectionable content (binary)
# y: [0, 1]
# words: the actual text
# cap_words: captions generated before (descriptions not transcriptions)
# cap index: tokenized cap_words
# each of the four categories

# NOTES
# all outputs are of form [x, y] where one of x or y is 1 and the other is 0 (why this rather than one label I don't know, whatever)