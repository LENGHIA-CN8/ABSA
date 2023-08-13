import torch
import numpy as np
from underthesea import word_tokenize
import re
from explore_data.data_utils import remove_emoji, word_segment, remove_special_character, remove_urls
from loader.load_acd_acsa import get_examples, convert_examples_to_dataset, aspect_type_dict, aspect_sentiment_dict
from constant import label_weights_acd_acsa, label_weights_imb_acd_acsa
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Constant parameter
DEVICE = "cuda"
TERM_VOCAB_FILE = "./resource/FMCG_term_vocab.txt"
NOT_TERM_FILE = "./resource/FMCG_not_term.txt"
DATA_MODE = "single"
STRIDE = 4

MODEL_CONFIG = {
  "MODEL_TYPE": "phobert",
  "MODEL_PATH_OR_TYPE": "vinai/phobert-base",
  "TRAINED_MODEL_PATH": "../saved_model/ACD_ACSA/phobert_seed_21_num_classes_4_max_length_256_stride_4_epochs_10_data_mode_single_pool_type_max_lr_2e-05_weight_decay_0.001_do_lower_case_True_replace_term_with_mask_True_imb_weight_False_train_batch_size_8_test_batch_size_8.pt",
  "DO_LOWER_CASE": True,
  "POOL_TYPE": "max",
  "NUM_CLASSES": 4,
  "OUTPUT_HIDDEN_STATES": True
}

MAX_LENGTH = 256
BATCH_SIZE = 4
REPLACE_TERM_WITH_MASK = True

if MODEL_CONFIG["NUM_CLASSES"] == 4:
    aspect_sentiment_dict.pop("conflict", None)

ASPECT_SENTIMENT_DICT = aspect_sentiment_dict
ASPECT_TYPE_DICT = aspect_type_dict

REVERSE_ASPECT_TYPE_DICT = dict([(idx, label) for (label, idx) in ASPECT_TYPE_DICT.items()])
REVERSE_ASPECT_SENTIMENT_DICT = dict([(idx, label) for (label, idx) in ASPECT_SENTIMENT_DICT.items()])
LABEL_WEIGHTS = torch.Tensor(label_weights_acd_acsa[:MODEL_CONFIG["NUM_CLASSES"] + 1]).to(DEVICE)


def preprocess(raw_sentence, term_vocab):
    # term vocab should be sorted by length (long to short)
    term_vocab = sorted(term_vocab, key= lambda term: len(term), reverse = True)

    sentence = " ".join(raw_sentence.split()) # fix white space

    # to remain the term unchange and avoid mistake when apply word segmentation,
    # we replace term with special tokens first and then replace back later

    sent_term_found_dict = {}
    pos_covered = [] # used to avoid term covers term

    for i, term in enumerate(term_vocab):
        for term_found in re.finditer(term, sentence.lower()):
            term_found_pos = term_found.start()
            if term_found_pos == -1:
                break

            covered = False
            for pos in pos_covered:
                if ((term_found_pos >= pos[0] and term_found_pos < pos[1]) or 
                    (term_found_pos + len(term) >= pos[0] and term_found_pos + len(term) < pos[1]) or 
                    (term_found_pos <= pos[0] and term_found_pos + len(term) >= pos[1])):
                    covered = True
                    break
            
            if not covered:
                pos_covered.append((term_found_pos, term_found_pos + len(term)))
                assert sentence[term_found_pos: term_found_pos + len(term)].lower() == term, "{} || {}".format(sentence[term_found_pos: term_found_pos + len(term)].lower(), term)
                sentence = sentence[:term_found_pos] + term.replace(" ", "_") + sentence[term_found_pos + len(term):]
                sent_term_found_dict["term{}".format(i)] = term.replace(" ", "_")
    
    for term_id in sent_term_found_dict:
        sentence = sentence.replace(sent_term_found_dict[term_id], " " + term_id + " ")

    # word segment, remove urls, emoji, special character
    sentence = word_segment(sentence)
    sentence = remove_urls(sentence)
    sentence = remove_emoji(sentence)
    sentence = remove_special_character(sentence)

    data_acd_acsa_format = {}
    data_acd_acsa_format["terms"] = {}
    # replace term back:
    for term_id in sent_term_found_dict:
        sentence = sentence.replace(term_id, sent_term_found_dict[term_id])

    tokens = sentence.split()
    for term_id in sent_term_found_dict:
        data_acd_acsa_format["terms"][term_id] = {"term": [], "attribute": []}
        for token_id, token in enumerate(tokens):
            if token == sent_term_found_dict[term_id]:
                data_acd_acsa_format["terms"][term_id]['term'].append([sent_term_found_dict[term_id], token_id])

    data_acd_acsa_format["tokens"] = tokens
    return data_acd_acsa_format

def get_acd_acsa_phobert_model(model_config):
    from transformers import PhobertTokenizer, RobertaConfig
    from model.ACD_ACSA_PhoBERT import Net

    config_class, tokenizer_class, model_class = RobertaConfig ,PhobertTokenizer, Net

    tokenizer = tokenizer_class.from_pretrained(model_config['MODEL_PATH_OR_TYPE'])
    tokenizer.do_lower_case = model_config["DO_LOWER_CASE"]
    config = config_class.from_pretrained(model_config['MODEL_PATH_OR_TYPE'], num_labels= model_config["NUM_CLASSES"], output_hidden_states= model_config["OUTPUT_HIDDEN_STATES"])
    model = model_class(model_config['MODEL_PATH_OR_TYPE'], config= config, num_classes= model_config["NUM_CLASSES"], num_categories= len(ASPECT_TYPE_DICT))
    model.load_state_dict(torch.load(model_config["TRAINED_MODEL_PATH"], map_location= torch.device(DEVICE)))
    model.to(DEVICE)
    return tokenizer, model

def main(input_data, models):
    with open(TERM_VOCAB_FILE, "r") as f:
        term_vocab = [term.replace("\n", "") for term in f.readlines()]
    with open(NOT_TERM_FILE, "r") as f:
        not_term = [term.replace("\n", "") for term in f.readlines()]
    term_vocab = set(term_vocab) - set(not_term)

    process_data = []
    for data in input_data:
        process_data.append(preprocess(data, term_vocab))

    tokenizer, model = models
    examples = get_examples(datapath= None, data= process_data, predict= True)
    dataset = convert_examples_to_dataset(examples, ASPECT_TYPE_DICT, ASPECT_SENTIMENT_DICT, MAX_LENGTH, tokenizer, STRIDE, REPLACE_TERM_WITH_MASK, predict= True)
    Sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=Sampler, batch_size= BATCH_SIZE)
    
    predictions = []
    aspect_type = []
    for batch in tqdm(dataloader, desc= "Predict"):
        model.eval()
        batch = [t.to(DEVICE) for t in batch]
        
        input_ids, attention_mask, token_type_ids, target_masks, aspect_type_labels = batch
        logits = model(input_ids, attention_mask, target_masks, aspect_type_labels, None, 
                                      DATA_MODE, MODEL_CONFIG["POOL_TYPE"], LABEL_WEIGHTS)


        out_prob = F.softmax(logits.view(-1, MODEL_CONFIG["NUM_CLASSES"]), 1)
        out_prob = out_prob.detach().cpu().numpy()
        predictions.extend(np.argmax(out_prob, axis=1))
        aspect_type.extend(aspect_type_labels.detach().cpu().numpy())

    predictions = [REVERSE_ASPECT_SENTIMENT_DICT[label_id] for label_id in predictions]
    aspect_type = [REVERSE_ASPECT_TYPE_DICT[type_id] for type_id in aspect_type]
    
    num_aspect_type = len(ASPECT_TYPE_DICT)
    total_cases = int(len(predictions)/num_aspect_type)
    assert total_cases == len(examples)

    results = {"results": []}
    for i in range(total_cases):
        sent_id = examples[i].sent_id
        if sent_id >= len(results["results"]):
            for not_related_data in input_data[len(results["results"]):sent_id]:
                results["results"].append({"text": not_related_data, "predictions": []})       
            result = {"text": input_data[sent_id], "predictions": []}
            results["results"].append(result)

        predict = {"term": examples[i].term.replace("_", " "), "aspects": {}}
        for j in range(num_aspect_type):
            if predictions[i*num_aspect_type + j] != "None":
                predict["aspects"][aspect_type[i*num_aspect_type + j]] = predictions[i*num_aspect_type + j]
        
        if predict["term"] == "EOS" and len(predict["aspects"]) == 0:
            continue
        results["results"][sent_id]["predictions"].append(predict)
    
    if len(results["results"]) < len(input_data):
        for not_related_data in input_data[len(results["results"]):]:
            results["results"].append({"text": not_related_data, "predictions": []})
    return results