import math

# label_dict = {"positive": 0, "neutral": 1, "negative": 2, "conflict": 3}

total_data = 9210
positive, neutral, negative, conflict = 4007, 4880, 269, 54

label_weights_tbsa = [1.0, 1.0, 1.0, 1.0]
label_weights_imb_tbsa = [math.log(total_data/positive), 
                          math.log(total_data/neutral), 
                          math.log(total_data/negative), 
                          math.log(total_data/conflict)]

label_weights_acd_acsa = [1.0, 1.0, 1.0, 1.0, 1.0]
label_weights_imb_acd_acsa = [1.0, 1.0, 1.0, 1.0, 1.0]

MODEL_FILE = {
    'phobert': {'config_file': "vinai/phobert-base",
                'model_file': "vinai/phobert-base",
                'vocab_file': None,
                'merges_file': None},
    'phobert_large': {"model_file": "vinai/phobert-large",
               "config_file": "vinai/phobert-large",
               "merges_file": None,
               "vocab_file": None},
    'xlm_roberta': {"model_file": "xlm-roberta-base",
                    "config_file": "xlm-roberta-base",
                    "merges_file": None,
                    "vocab_file": None},
    'xlm_roberta_large': {"model_file": "xlm-roberta-large",
                    "config_file": "xlm-roberta-large",
                    "merges_file": None,
                    "vocab_file": None},
    'vibert': {"model_file": "FPTAI/vibert-base-cased",
               "config_file": "FPTAI/vibert-base-cased",
               "merges_file": None,
               "vocab_file": None}
}