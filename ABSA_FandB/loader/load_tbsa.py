import json
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
import os

label_dict = {"positive": 0, "neutral": 1, "negative": 2, "conflict": 3}

class Example:
    def __init__(self, tokens: list, term: str, position: list, label: str, sent_id: int):
        self.sent_id = sent_id
        self.tokens = tokens
        self.term = term
        self.position = position
        self.label = label
        self.predict = None

class Upsampling:
    def __init__(self, label, xtime):
        self.label = label
        self.xtime = xtime

# def get_examples(datapath, get_conflict= False, upsampling= None):
#     """
#     :param:
#       - datapath: data file path (cleaned)
#     :return:
#       - list of examples for TBSA task

#     Each distinct term of sentence makes one example
#     """

#     with open(datapath, "r") as f:
#         data = json.load(f)

#     examples = []
#     term_with_no_attribute = 0
#     for doc_id in data:
#         for sentence in data[doc_id]:
#             tokens = sentence["tokens"]
#             for term_id in sentence['terms']:
#                 # Since a term can appear multiple time in a sentence and the text 
#                 # can be different in different position. We use the first text to
#                 # indentify that term and replace in tokens.
#                 term_text = sentence['terms'][term_id]['term'][0][0]

#                 if term_text == "EOS": # We don't want to consider general sentiment
#                     continue           # for sentence in TBSA task (only use in ACSA)
                
#                 term_position = []
#                 term_label = None

#                 sentence['terms'][term_id]['term'] = sorted(sentence['terms'][term_id]['term'], key= lambda x: x[1])

#                 previous_term_pos = -1e3 # Sometime coref terms will appear right next to the previous
#                                          # term and do not meaningful. So we will remove the repeat term
#                                          # if their positions are consecutive.          
#                 term_pos = [x[1] for x in sentence['terms'][term_id]['term']]
#                 new_tokens = []
#                 for i, token in enumerate(tokens):
#                     if i in term_pos:
#                         if i == previous_term_pos + 1:
#                             continue
#                         else: 
#                             term_position.append(len(new_tokens))
#                             new_tokens.append(term_text)
#                         previous_term_pos = i
#                     else:
#                         new_tokens.append(token)

                     
#                 sentiment = set([x[1] for x in sentence['terms'][term_id]['attribute']])
                
#                 if "None" in sentiment:   # We do not consider the terms which is unclear 
#                     continue              # (is in consideration of human)

#                 # if term contains conflict or both positive and negative sentiment
#                 # -> conflict
#                 # if term contains positive and neutral (or negative and neutral)
#                 # -> positive (or negative)
#                 if len(sentiment) == 1:
#                     term_label = sentiment.pop()
#                 elif 'conflict' in sentiment or ('positive' in sentiment and 'negative' in sentiment):
#                     term_label = "conflict"
#                 elif 'positive' in sentiment:
#                     term_label = "positive"
#                 elif 'negative' in sentiment:
#                     term_label = "negative"
#                 else:
#                     print('Can not extract sentiment label from {}'.format(sentiment))
#                     term_with_no_attribute += 1
#                     continue

#                 if not get_conflict and term_label == "conflict": # get conflict term or not
#                     continue

#                 example = Example(tokens= new_tokens, term= term_text, position= term_position, label= term_label)
#                 if upsampling is not None:
#                     if term_label == upsampling.label:
#                         examples.extend([example]* upsampling.xtime)
#                     else:
#                         examples.append(example)
#                 else:
#                     examples.append(example)
#     print("Term with no attribute:", term_with_no_attribute)          
#     for example in examples[:1]:
#         print("Tokens:", example.tokens)
#         print("Term:", example.term)
#         print("Position:", example.position)
#         print("Sentiment:", example.label)
#         print("="*100)
#     return examples

def get_examples(datapath, get_conflict= False, upsampling= None, data= None):
    """
    :param:
      - datapath: data file path (cleaned + reformat to tbsa)
    :return:
      - list of examples for TBSA task

    Each distinct term of sentence makes one example
    """

    if data is None:
        with open(datapath, "r") as f:
            data = json.load(f)['data']

    examples = []

    for sent_id, sentence in enumerate(data):
        tokens = sentence["tokens"]
        for term_id in sentence['terms']:
            # Since a term can appear multiple time in a sentence and the text 
            # can be different at different position. We use the first text to
            # indentify that term and replace for all tokens mentioning that term.
            term_text = sentence['terms'][term_id]['term'][0][0]

            if term_text == "EOS": # We don't want to consider general sentiment
                continue           # for sentence in TBSA task (only use in ACSA)
            
            term_position = []
            term_label = None

            sentence['terms'][term_id]['term'] = sorted(sentence['terms'][term_id]['term'], key= lambda x: x[1])

            previous_term_pos = -1e3  # Sometime coref terms will appear right next to the previous
                                      # term and do not meaningful. So we will remove the repeat term
                                      # if their positions are consecutive.          
            term_pos = [x[1] for x in sentence['terms'][term_id]['term']]
            new_tokens = []
            for token_pos, token in enumerate(tokens):
                if token_pos in term_pos:
                    if token_pos == previous_term_pos + 1:
                        continue
                    else: 
                        term_position.append(len(new_tokens))
                        new_tokens.append(term_text)
                    previous_term_pos = token_pos
                else:
                    new_tokens.append(token)
          
            term_label = sentence['terms'][term_id]['sentiment']

            if not get_conflict and term_label == "conflict": # get conflict term or not
                continue

            example = Example(tokens= new_tokens, term= term_text, position= term_position, label= term_label, sent_id= sent_id)
            if upsampling is not None:
                if term_label == upsampling.label:
                    examples.extend([example]* upsampling.xtime)
                else:
                    examples.append(example)
            else:
                examples.append(example)
         
    # for example in examples[:1]:
    #     print("Tokens:", example.tokens)
    #     print("Term:", example.term)
    #     print("Position:", example.position)
    #     print("Sentiment:", example.label)
    #     print("="*100)
    return examples

def convert_examples_to_dataset(examples, label_dict, seq_max_len, tokenizer, stride, replace_term_with_mask= True, return_dataset= 'pt', predict= False):
    """
    replace_term_with_mask: the sentiment of a term mostly do not depend on it text but
    depend on the context of it. So we might want to remove the text of the term with <mask>
    token to reduce complicated in process term text. Moreover some terms may appear a lot
    with the same sentiment in the corpus (i.e. Comfort is mostly positive), this might lead
    to only a class sentiment prediction is make for that term.

    :return:
        input_ids
        attention_mask
        token_type_ids
        target_masks
        labels
    """
    
    input_ids = []
    attention_masks = []
    token_type_ids = []
    target_masks = []
    labels = []

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    mask_token = tokenizer.mask_token

    for example_id, example in enumerate(examples):
        best_tokens = []
        best_target_mask = []
        best_attention_mask = []
        best_token_type_id = []
        best_score = -1e-3

        for start_id in range(0, len(example.tokens), stride):
            tokens = []
            target_mask = []
            
            visited_all_token =  True
            for i, token in enumerate(example.tokens[start_id:]):
                if (i + start_id) in example.position:
                    if replace_term_with_mask:
                        tokens.append(mask_token)
                        target_mask.append(1)
                    else:
                        subtokens = tokenizer.tokenize(token)
                        tokens.extend(subtokens)
                        target_mask.extend([1] * len(subtokens))
                else:
                    subtokens = tokenizer.tokenize(token)
                    tokens.extend(subtokens)
                    target_mask.extend([0] * len(subtokens))

                if len(tokens) >= seq_max_len - 2:
                    end_id = i + start_id
                    visited_all_token = False
                    break

                end_id = i + start_id  

            score = calculate_term_context_score(start_id, end_id, example.position)
            if score > best_score:
                best_score = score
                best_tokens = tokens
                best_target_mask = target_mask
              
            if visited_all_token:
                break

        # truncation
        best_tokens = best_tokens[:seq_max_len - 2] # placehold for cls and sep tokens
        best_target_mask = best_target_mask[:seq_max_len - 2]
        
        best_tokens = [cls_token] + best_tokens + [sep_token]
        best_target_mask = [0] + best_target_mask + [0]
        best_attention_mask = [1] * len(best_tokens)
        best_token_type_id = [0] * len(best_tokens)

        # padding
        if len(tokens) < seq_max_len:
            zero_padding = [0] * (seq_max_len - len(best_tokens))
            one_padding = [1] * (seq_max_len - len(best_tokens))

            best_tokens += [pad_token] * (seq_max_len - len(best_tokens))
            best_target_mask += zero_padding
            best_attention_mask += zero_padding
            best_token_type_id += one_padding

        best_input_id = tokenizer.convert_tokens_to_ids(best_tokens)
        assert len(best_input_id) == seq_max_len, "Length is not equal {} vs {}".format(len(best_input_id), seq_max_len)
        assert len(best_target_mask) == seq_max_len, "Length is not equal {} vs {}".format(len(best_target_mask), seq_max_len)
        assert len(best_attention_mask) == seq_max_len, "Length is not equal {} vs {}".format(len(best_attention_mask), seq_max_len)
        assert len(best_token_type_id) == seq_max_len, "Length is not equal {} vs {}".format(len(best_token_type_id), seq_max_len)

        input_ids.append(best_input_id)
        target_masks.append(best_target_mask)
        attention_masks.append(best_attention_mask)
        token_type_ids.append(best_token_type_id)
        if not predict:
            labels.append(label_dict[example.label])

    # for i in range(len(input_ids)):
    #     if sum(target_masks[i]) != len(examples[i].position):
    #         print(sum(target_masks[i]), len(examples[i].position))
    #         print(input_ids[i])
    #         print(tokenizer.convert_ids_to_tokens(input_ids[i]))
    #         print(target_masks[i])
    #         if not predict:
    #             print(labels[i])
    #     if sum(target_masks[i]) == 0:
    #         print("Somthing wrong" * 50)

    # for i in range(2):
    #     print(input_ids[i])
    #     print(tokenizer.convert_ids_to_tokens(input_ids[i]))
    #     print(target_masks[i])
    #     if not predict:
    #         print(labels[i])
    
    # info
    # number_of_terms = 0
    # number_of_extracted_terms = 0
    # for target_mask in target_masks:
    #     for i in range(seq_max_len):
    #         if target_mask[i] == 1:
    #             if target_mask[i-1] == 0 or i == 0:
    #                 number_of_extracted_terms += 1

    # for example in examples:
    #    number_of_terms += len(example.position)

    # print("Number of all terms:", number_of_terms)
    # print("Number of all extracted terms:", number_of_extracted_terms)
    # print("Number of distinct terms:", len(examples))

    if return_dataset == "pt":
        all_input_ids = torch.tensor([input_id for input_id in input_ids], dtype=torch.long)
        all_attention_masks = torch.tensor([attention_mask for attention_mask in attention_masks], dtype=torch.long)
        all_token_type_ids = torch.tensor([token_type_id for token_type_id in token_type_ids], dtype=torch.long)
        all_target_masks = torch.tensor([target_mask for target_mask in target_masks], dtype=torch.long)
        if not predict:
            all_labels = torch.tensor([label for label in labels], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_target_masks, all_labels)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_target_masks)
        return dataset
        
    if not predict:
        return input_ids, attention_masks, token_type_ids, target_masks, labels
    else:
        return input_ids, attention_masks, token_type_ids, target_masks


def calculate_term_context_score(start_token_id, end_token_id, term_position):
    term_inside = []
    for term_pos in term_position:
        if start_token_id <= term_pos <= end_token_id:
            term_inside.append(term_pos)
    if len(term_inside) == 0:
      return -1
    left_context = min(term_inside) - start_token_id
    right_context = end_token_id - max(term_inside)
    score = 0.2 * min(left_context, right_context)/(end_token_id - start_token_id) + 0.8 * len(term_inside) / len(term_position)
    return score


class Loader():
    def __init__(self, args, label_dict, tokenizer):
        self.args = args
        self.train_file = os.path.join(args.data_dir, "train.json")
        self.dev_file = os.path.join(args.data_dir, "dev.json")
        self.test_file = os.path.join(args.data_dir, "test.json")
        self.label_dict = label_dict
        self.tokenizer = tokenizer

    def load_data(self):
        print("Load training data ...")
        train_examples = get_examples(self.train_file, self.args.get_conflict)
        train_data = convert_examples_to_dataset(train_examples, self.label_dict, self.args.max_length, self.tokenizer, self.args.replace_term_with_mask)
        Sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=Sampler, batch_size= self.args.train_batch_size)

        print("Load dev data ...")
        dev_examples  = get_examples(self.dev_file, self.args.get_conflict)
        dev_data = convert_examples_to_dataset(dev_examples, self.label_dict, self.args.max_length, self.tokenizer, self.args.replace_term_with_mask)
        Sampler = SequentialSampler(dev_data)
        dev_loader = DataLoader(dev_data, sampler=Sampler, batch_size= self.args.test_batch_size)

        print("Load test data ...")
        test_examples = get_examples(self.test_file, self.args.get_conflict)
        test_data = convert_examples_to_dataset(test_examples, self.label_dict, self.args.max_length, self.tokenizer, self.args.replace_term_with_mask)
        Sampler = SequentialSampler(test_data)
        test_loader = DataLoader(test_data, sampler=Sampler, batch_size= self.args.test_batch_size)

        return train_loader, dev_loader, test_loader
