import os
import argparse
import json
import random



def convert_orig_to_tbsa_format(sentence):
    new_sentence = {}
    new_sentence["tokens"] = sentence["tokens"]
  
    new_sentence["terms"] = {}
    term_with_no_attribute = 0
    for term_id in sentence['terms']:
        sentiment = set([x[1] for x in sentence['terms'][term_id]['attribute']])
        
        if "None" in sentiment:   # We do not consider the terms which is unclear 
            continue              # (is in consideration of human)

        # if term contains conflict or both positive and negative sentiment
        # -> conflict
        # if term contains positive and neutral (or negative and neutral)
        # -> positive (or negative)
        if len(sentiment) == 1:
            term_label = sentiment.pop()
        elif 'conflict' in sentiment or ('positive' in sentiment and 'negative' in sentiment):
            term_label = "conflict"
        elif 'positive' in sentiment:
            term_label = "positive"
        elif 'negative' in sentiment:
            term_label = "negative"
        else:
            print('Can not extract sentiment label from {}'.format(sentiment)) # Human mistake here
            term_with_no_attribute += 1                                        # forgot to annotate a term
            continue

        new_sentence["terms"][term_id] = {}
        new_sentence["terms"][term_id]['term'] = sentence['terms'][term_id]['term']
        new_sentence["terms"][term_id]['sentiment'] = term_label
    
    return new_sentence, term_with_no_attribute

        

parser = argparse.ArgumentParser(description='split TBSA')
parser.add_argument('--data_file', type=str, required= True, help='processed data file')
parser.add_argument('--save_folder', type=str, required= True, help='save folder')
args = parser.parse_args()


with open(args.data_file, "r") as f:
    data = json.load(f)

all_sentences = []
term_with_no_attribute = 0
for doc_id in data:
    for sentence in data[doc_id]:
        coverted_info = convert_orig_to_tbsa_format(sentence)
        convert_sent = coverted_info[0]
        term_with_no_attribute += coverted_info[1]
        if len(sentence["terms"]) > 0:
            all_sentences.append(convert_sent)
print("Term with no attribute:", term_with_no_attribute)
all_data_tbsa = {}
all_data_tbsa = {"version": args.data_file[args.data_file.index("ver"): args.data_file.index("ver") + 6], "data": all_sentences}

# Save to convert file
with open(os.path.join(args.save_folder, "all_tbsa.json"), "w") as f:
    json.dump(all_data_tbsa, f, indent = 4)


# Because negative label is minority so we split seperately for this label
sent_contain_negative = []
sent_contain_positive_and_neutral = []

for sentence in all_sentences:
    sentiments = []
    for term_id in sentence['terms']:
        sentiments.append(sentence['terms'][term_id]['sentiment'])
    if "negative" in sentiments:
        sent_contain_negative.append(sentence)
    else:
        sent_contain_positive_and_neutral.append(sentence)

# split train, dev, test
random.shuffle(sent_contain_negative)
random.shuffle(sent_contain_positive_and_neutral)

train_set, dev_set, test_set = [], [], []
train_rate, dev_rate, test_rate = 0.8, 0.1, 0.1


num_negative = len(sent_contain_negative)
num_positive_neutral = len(sent_contain_positive_and_neutral)

train_set = sent_contain_negative[:int(train_rate*num_negative)] + sent_contain_positive_and_neutral[:int(train_rate*num_positive_neutral)]
dev_set = sent_contain_negative[int(train_rate*num_negative):int(train_rate*num_negative)+ int(dev_rate*num_negative)] + sent_contain_positive_and_neutral[int(train_rate*num_positive_neutral):int(train_rate*num_positive_neutral)+ int(dev_rate*num_positive_neutral)]
test_set = sent_contain_negative[int(train_rate*num_negative)+ int(dev_rate*num_negative):] + sent_contain_positive_and_neutral[int(train_rate*num_positive_neutral)+ int(dev_rate*num_positive_neutral):]
print("All sentences:", len(all_sentences))
print("Train: {} || Dev: {} || Test: {}".format(len(train_set), len(dev_set), len(test_set)))

train_data, dev_data, test_data = {"version":all_data_tbsa["version"]}, {"version":all_data_tbsa["version"]}, {"version":all_data_tbsa["version"]}
train_data["data"], dev_data["data"], test_data["data"] = train_set, dev_set, test_set

with open(os.path.join(args.save_folder, "train.json"), "w") as f:
    json.dump(train_data, f, indent = 4)

with open(os.path.join(args.save_folder, "dev.json"), "w") as f:
    json.dump(dev_data, f, indent = 4)

with open(os.path.join(args.save_folder, "test.json"), "w") as f:
    json.dump(test_data, f, indent = 4)

