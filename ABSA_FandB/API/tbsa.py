import torch
import numpy as np
from underthesea import word_tokenize
import re
from explore_data.data_utils import remove_emoji, word_segment, remove_special_character, remove_urls
from loader.load_tbsa import get_examples, convert_examples_to_dataset, label_dict
from constant import label_weights_tbsa, label_weights_imb_tbsa
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Constant parameter
DEVICE = "cuda"
TERM_VOCAB_FILE = "./resource/FandB_term_vocab.txt"
NOT_TERM_FILE = "./resource/FandB_not_term.txt"
DATA_MODE = "single"
STRIDE = 4

MODEL_CONFIG = {
  "MODEL_TYPE": "phobert_mixlayer",
  "MODEL_PATH_OR_TYPE": "vinai/phobert-base",
  "TRAINED_MODEL_PATH": "../saved_model/TBSA/phobert_mixlayer_seed_21_num_classes_3_max_length_256_stride_4_epochs_10_data_mode_single_pool_type_max_lr_3e-05_weight_decay_0.001_do_lower_case_True_replace_term_with_mask_True_imb_weight_True_mix_count_4_mix_type_HSUM_train_batch_size_8.pt",
  "DO_LOWER_CASE": True,
  "MIX_COUNT": 4,
  "MIX_TYPE": "HSUM",
  "POOL_TYPE": "max",
  "NUM_CLASSES": 3,
  "OUTPUT_HIDDEN_STATES": True
}

MAX_LENGTH = 256
BATCH_SIZE = 4
REPLACE_TERM_WITH_MASK = True

if MODEL_CONFIG["NUM_CLASSES"] == 3:
    label_dict.pop("conflict", None)
LABEL_DICT = label_dict

REVERSE_LABEL_DICT = dict([(idx, label) for (label, idx) in LABEL_DICT.items()])
LABEL_WEIGHTS = torch.Tensor(label_weights_imb_tbsa[:MODEL_CONFIG["NUM_CLASSES"] + 1]).to(DEVICE)


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

    data_tbsa_format = {}
    data_tbsa_format["terms"] = {}
    # replace term back:
    for term_id in sent_term_found_dict:
        sentence = sentence.replace(term_id, sent_term_found_dict[term_id])

    tokens = sentence.split()
    for term_id in sent_term_found_dict:
        data_tbsa_format["terms"][term_id] = {"term": [], "sentiment": None}
        for token_id, token in enumerate(tokens):
            if token == sent_term_found_dict[term_id]:
                data_tbsa_format["terms"][term_id]['term'].append([sent_term_found_dict[term_id], token_id])

    data_tbsa_format["tokens"] = tokens
    return data_tbsa_format

def get_tbsa_phobert_model(model_config):
    from transformers import PhobertTokenizer, RobertaConfig
    from model.TBSA_PhoBERT_MixLayers import Net

    config_class, tokenizer_class, model_class = RobertaConfig ,PhobertTokenizer, Net

    tokenizer = tokenizer_class.from_pretrained(model_config['MODEL_PATH_OR_TYPE'])
    tokenizer.do_lower_case = model_config["DO_LOWER_CASE"]
    config = config_class.from_pretrained(model_config['MODEL_PATH_OR_TYPE'], num_labels= model_config["NUM_CLASSES"], output_hidden_states= model_config["OUTPUT_HIDDEN_STATES"])
    model = model_class(model_config['MODEL_PATH_OR_TYPE'], config= config, num_classes= model_config["NUM_CLASSES"], count= model_config["MIX_COUNT"], mix_type= model_config["MIX_TYPE"])
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
    examples = get_examples(datapath= None, data= process_data)
    dataset = convert_examples_to_dataset(examples, LABEL_DICT, MAX_LENGTH, tokenizer, STRIDE, REPLACE_TERM_WITH_MASK, predict= True)
    Sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=Sampler, batch_size= BATCH_SIZE)
    
    predictions = []
    for batch in tqdm(dataloader, desc= "Predict"):
        model.eval()
        batch = [t.to(DEVICE) for t in batch]
        input_ids, attention_mask, token_type_ids, target_masks = batch
        logits = model(input_ids, attention_mask, target_masks, None, DATA_MODE, MODEL_CONFIG["POOL_TYPE"], LABEL_WEIGHTS)

        out_prob = F.softmax(logits.view(-1, MODEL_CONFIG["NUM_CLASSES"]), 1)
        out_prob = out_prob.detach().cpu().numpy()
        predictions.extend(np.argmax(out_prob, axis=1))

    predictions = [REVERSE_LABEL_DICT[label_id] for label_id in predictions]
    
    results = {"results": []}
    for i, prediction in enumerate(predictions):
        sent_id = examples[i].sent_id

        if sent_id >= len(results["results"]):
            for not_related_data in input_data[len(results["results"]):sent_id]:
                results["results"].append({"text": not_related_data, "predictions": []})
            
            result = {"text": input_data[sent_id], "predictions": []}
            results["results"].append(result)

        results["results"][sent_id]["predictions"].append({"term": examples[i].term.replace("_", " "), "sentiment": prediction})
    
    if len(results["results"]) < len(input_data):
        for not_related_data in input_data[len(results["results"]):]:
            results["results"].append({"text": not_related_data, "predictions": []})
    return results
