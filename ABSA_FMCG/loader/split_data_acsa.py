import os
import argparse
import json
import random



def convert_orig_to_acsa_format(sentence):
    new_sentence = {}
    new_sentence["tokens"] = sentence["tokens"]
  
    new_sentence["terms"] = {}
    term_with_no_attribute = 0
    for term_id in sentence['terms']:
        sentiment = set([x[1] for x in sentence['terms'][term_id]['attribute']])
        
        if "None" in sentiment:   # We do not consider the terms which is unclear 
            continue              # (is in consideration of human)

        if len(sentiment) < 1:
            term_with_no_attribute += 1 # Check human error
            continue

        new_sentence["terms"][term_id] = {}
        new_sentence["terms"][term_id]['term'] = sentence['terms'][term_id]['term']
        new_sentence["terms"][term_id]['attribute'] = sentence['terms'][term_id]['attribute']
    
    return new_sentence, term_with_no_attribute

parser = argparse.ArgumentParser(description='split ACSA')
parser.add_argument('--data_file', type=str, required= True, help='processed data file')
parser.add_argument('--save_folder', type=str, required= True, help='save folder')
args = parser.parse_args()


with open(args.data_file, "r") as f:
    data = json.load(f)

all_sentences = []
term_with_no_attribute = 0
for doc_id in data:
    for sentence in data[doc_id]:
        coverted_info = convert_orig_to_acsa_format(sentence)
        convert_sent = coverted_info[0]
        term_with_no_attribute += coverted_info[1]
        if len(sentence["terms"]) > 0:
            all_sentences.append(convert_sent)
print("Term with no attribute:", term_with_no_attribute)
all_data_tbsa = {}
all_data_tbsa = {"version": args.data_file[args.data_file.index("ver"): args.data_file.index("ver") + 6], "data": all_sentences}


# Save to convert file
if not os.path.isdir(args.save_folder):
    os.mkdir(args.save_folder)
with open(os.path.join(args.save_folder, "all_acsa.json"), "w") as f:
    json.dump(all_data_tbsa, f, indent = 4)


# Because service category is minority so we split seperately for this category
sent_contain_service = []
sent_contain_others = []

for sentence in all_sentences:
    categories = []
    for term_id in sentence['terms']:
        categories.extend([x[0] for x in sentence['terms'][term_id]['attribute']])
    if "service" in categories:
        sent_contain_service.append(sentence)
    else:
        sent_contain_others.append(sentence)

# split train, dev, test
random.shuffle(sent_contain_service)
random.shuffle(sent_contain_others)

train_set, dev_set, test_set = [], [], []
train_rate, dev_rate, test_rate = 0.8, 0.1, 0.1


num_service = len(sent_contain_service)
num_others = len(sent_contain_others)

train_set = sent_contain_service[:int(train_rate*num_service)] + sent_contain_others[:int(train_rate*num_others)]
dev_set = sent_contain_service[int(train_rate*num_service):int(train_rate*num_service)+ int(dev_rate*num_service)] + sent_contain_others[int(train_rate*num_others):int(train_rate*num_others)+ int(dev_rate*num_others)]
test_set = sent_contain_service[int(train_rate*num_service)+ int(dev_rate*num_service):] + sent_contain_others[int(train_rate*num_others)+ int(dev_rate*num_others):]
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

