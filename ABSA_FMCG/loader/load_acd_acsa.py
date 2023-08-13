import json
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
import os


aspect_type_dict = {"general": 0, "quality": 1, "price": 2, "service": 3}
aspect_sentiment_dict = {"None": 0, "positive": 1, "neutral": 2, "negative": 3, "conflict": 4}

class Example:
    def __init__(self, tokens: list, term: str, position: list, aspects: list, sent_id: int):
        self.sent_id = sent_id
        self.tokens = tokens
        self.term = term
        self.position = position
        self.aspects = aspects
        self.predict_type = list()
        self.predict_sentiment = list()

def get_examples(datapath, get_conflict= False, upsampling= None, data= None, predict= False):
    """
    :param:
      - datapath: data file path (cleaned + reformat to acsa)
    :return:
      - list of examples for ACSA task

    Each distinct term of sentence makes one example
    """

    if data is None:
        with open(datapath, "r") as f:
            data = json.load(f)['data']

    examples = []
    
    for sent_id, sentence in enumerate(data):
        contain_EOS = False

        tokens = sentence["tokens"]
        for term_id in sentence['terms']:
            # Since a term can appear multiple time in a sentence and the text 
            # can be different at different position. We use the first text to
            # indentify that term and replace for all tokens mentioning that term.
            term_text = sentence['terms'][term_id]['term'][0][0]
            term_position = []
            term_aspects = []

            sentence['terms'][term_id]['term'] = sorted(sentence['terms'][term_id]['term'], key= lambda x: x[1])

            previous_term_pos = -1e3  # Sometime coref terms will appear right next to the previous
                                      # term and do not meaningful. So we will remove the repeat term
                                      # if their positions are consecutive.          
            term_pos = [x[1] for x in sentence['terms'][term_id]['term']]
            new_tokens = []
            
            if term_text != "EOS":
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
            else:
                new_tokens = tokens
                
            for attribute in sentence['terms'][term_id]['attribute']:
                if not get_conflict and attribute[1] == "conflict":
                    continue
                term_aspects.append(attribute)

            if len(term_aspects) < 1 and not predict: # In traing phase we remove terms that do not contain aspect
                continue                              # (human mistake) or only contain conflict but get_conflict is False
            
            if term_text == "EOS":    # Can use for CLS token 
                term_position = [-1]  # in BERT-type model
                contain_EOS = True

            example = Example(tokens= new_tokens, term= term_text, position= term_position, aspects= term_aspects, sent_id= sent_id)
            examples.append(example)
        
        if not contain_EOS:                                                                                                    # Every sentences will be predicted in case
            example = Example(tokens= tokens, term= "EOS", position= [-1], aspects= [["general", "None"]], sent_id= sent_id)   # of general aspect for full sentence
            examples.append(example)
            
    # for example in examples[:1]:
    #     print("Tokens:", example.tokens)
    #     print("Term:", example.term)
    #     print("Position:", example.position)
    #     print("Aspects:", example.aspects)
    #     print("="*100)
    return examples        

def convert_examples_to_dataset(examples, aspect_type_dict, aspect_sentiment_dict, seq_max_len, tokenizer, stride, replace_term_with_mask= True, return_dataset= 'pt', predict= False):
    """
    replace_term_with_mask: the sentiment of a term mostly do not depend on it text but
    depend on the context of it. So we might want to replace the text of the term with <mask>
    token to reduce complicated in process term text. Moreover some terms may appear a lot
    with the same sentiment in the corpus (i.e. Comfort is mostly positive), this might lead
    to only a class sentiment prediction is make for that term.

    :return:
        input_ids
        attention_mask
        token_type_ids
        target_masks
        aspect_type_labels (padding with -1)
        aspect_sentiment_labels (padding with -1)
    """
    
    input_ids = []
    attention_masks = []
    token_type_ids = []
    target_masks = []
    aspect_type_labels = []
    aspect_sentiment_labels = []

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    mask_token = tokenizer.mask_token

    for example in examples:
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

            if -1 in example.position: # EOS token
                best_tokens = tokens
                best_target_mask = target_mask
                break

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
        if len(best_tokens) < seq_max_len:
            zero_padding = [0] * (seq_max_len - len(best_tokens))
            one_padding = [1] * (seq_max_len - len(best_tokens))

            best_tokens += [pad_token] * (seq_max_len - len(best_tokens))
            best_target_mask += zero_padding
            best_attention_mask += zero_padding
            best_token_type_id += one_padding
        
        if -1 in example.position: # EOS token
            assert len(example.position) == 1
            best_target_mask[0] = 1 # position for this term will be CLS token

        best_input_id = tokenizer.convert_tokens_to_ids(best_tokens)
        assert len(best_input_id) == seq_max_len, "Length is not equal {} vs {}".format(len(best_input_id), seq_max_len)
        assert len(best_target_mask) == seq_max_len, "Length is not equal {} vs {}".format(len(best_target_mask), seq_max_len)
        assert len(best_attention_mask) == seq_max_len, "Length is not equal {} vs {}".format(len(best_attention_mask), seq_max_len)
        assert len(best_token_type_id) == seq_max_len, "Length is not equal {} vs {}".format(len(best_token_type_id), seq_max_len)

        example_aspect_ids = {aspect_type_dict[aspect[0]]: aspect_sentiment_dict[aspect[1]] for aspect in example.aspects}
        for aspect_type_id in range(len(aspect_type_dict)):
            input_ids.append(best_input_id)
            target_masks.append(best_target_mask)
            attention_masks.append(best_attention_mask)
            token_type_ids.append(best_token_type_id)

            aspect_type_labels.append(aspect_type_id)

            if not predict:
                if aspect_type_id in example_aspect_ids:
                    aspect_sentiment_labels.append(example_aspect_ids[aspect_type_id])
                else:
                    aspect_sentiment_labels.append(aspect_sentiment_dict["None"])

    # for i in range(len(input_ids)):
    #     if sum(target_masks[i]) != len(examples[i//len(aspect_type_dict)].position):
    #         print(sum(target_masks[i]), "vs", examples[i//len(aspect_type_dict)].position)
    #         print(input_ids[i])
    #         print(tokenizer.convert_ids_to_tokens(input_ids[i]))
    #         print(target_masks[i])
    #         print(aspect_type_labels[i])
    #         if not predict:
    #             print(aspect_sentiment_labels[i])
    #     if sum(target_masks[i]) == 0:
    #       print("Something wrong" * 50)

    # for i in range(10): 
    #     print(input_ids[i*len(aspect_type_dict)])
    #     print(tokenizer.convert_ids_to_tokens(input_ids[i]))
    #     print(target_masks[i])
    #     print(aspect_type_labels[i])
    #     if not predict:
    #         print(aspect_sentiment_labels[i])
    
    for i in range(len(input_ids)):
        assert sum(target_masks[i]) != 0, "{}".format(sum(target_masks[i]))

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
    # print("Number of all extracted terms:", number_of_extracted_terms//len(aspect_type_dict))
    # print("Number of distinct terms:", len(examples))

    if return_dataset == "pt":
        all_input_ids = torch.tensor([input_id for input_id in input_ids], dtype=torch.long)
        all_attention_masks = torch.tensor([attention_mask for attention_mask in attention_masks], dtype=torch.long)
        all_token_type_ids = torch.tensor([token_type_id for token_type_id in token_type_ids], dtype=torch.long)
        all_target_masks = torch.tensor([target_mask for target_mask in target_masks], dtype=torch.long)
        all_aspect_type_labels = torch.tensor([label for label in aspect_type_labels], dtype=torch.long)
        if not predict:
            all_aspect_sentiment_labels = torch.tensor([label for label in aspect_sentiment_labels], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_target_masks, all_aspect_type_labels, all_aspect_sentiment_labels)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_target_masks, all_aspect_type_labels)
        return dataset

    if not predict:
        return input_ids, attention_masks, token_type_ids, target_masks, all_aspect_type_labels, all_aspect_sentiment_labels
    else:
        return input_ids, attention_masks, token_type_ids, target_masks, all_aspect_type_labels

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
    def __init__(self, args, aspect_type_dict, aspect_sentiment_dict, tokenizer):
        self.args = args
        self.train_file = os.path.join(args.data_dir, "train.json")
        self.dev_file = os.path.join(args.data_dir, "dev.json")
        self.test_file = os.path.join(args.data_dir, "test.json")
        self.aspect_type_dict = aspect_type_dict
        self.aspect_sentiment_dict = aspect_sentiment_dict
        self.tokenizer = tokenizer

    def load_data(self):
        print("Load training data ...")
        train_examples = get_examples(self.train_file, self.args.get_conflict)
        train_data = convert_examples_to_dataset(train_examples, self.aspect_type_dict, self.aspect_sentiment_dict, self.args.max_length, self.tokenizer, self.args.stride, self.args.replace_term_with_mask)
        Sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=Sampler, batch_size= self.args.train_batch_size)

        print("Load dev data ...")
        dev_examples  = get_examples(self.dev_file, self.args.get_conflict)
        dev_data = convert_examples_to_dataset(dev_examples, self.aspect_type_dict, self.aspect_sentiment_dict, self.args.max_length, self.tokenizer, self.args.stride, self.args.replace_term_with_mask)
        Sampler = SequentialSampler(dev_data)
        dev_loader = DataLoader(dev_data, sampler=Sampler, batch_size= self.args.test_batch_size)

        print("Load test data ...")
        test_examples = get_examples(self.test_file, self.args.get_conflict)
        test_data = convert_examples_to_dataset(test_examples, self.aspect_type_dict, self.aspect_sentiment_dict, self.args.max_length, self.tokenizer, self.args.stride, self.args.replace_term_with_mask)
        Sampler = SequentialSampler(test_data)
        test_loader = DataLoader(test_data, sampler=Sampler, batch_size= self.args.test_batch_size)

        return train_loader, dev_loader, test_loader