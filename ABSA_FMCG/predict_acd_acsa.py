from constant import label_weights_acd_acsa, label_weights_imb_acd_acsa, MODEL_FILE
from time import time
from utils import print_model_report
from transformers import PhobertTokenizer, RobertaConfig
import torch, random
from tqdm import tqdm
import numpy as np
import os
import argparse
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from loader.load_acd_acsa import convert_examples_to_dataset, get_examples
from evaluate.evaluate_acd_acsa import mi_ma_cro_f1, get_confusion_matrix, get_strict_acc_acd, get_strict_acc_acd_acsa, mi_ma_cro_f1_noNone

MODEL_CLASSES = {
}

def get_args():
    parser = argparse.ArgumentParser(description='ACD ACSA')
    # Arguments
    parser.add_argument('--task', type=str, choices= ["ACD_ACSA"], default='ACD_ACSA', 
                        help='ACD_ACSA')
    parser.add_argument('--approach', type=str, default='phobert', 
                        help='Model type')
    parser.add_argument("--predict_file", default=None, type=str, required=False,
                        help="predict file") 
    parser.add_argument("--model_path", default=None, type=str, required=False,
                        help="trained model path") 
    parser.add_argument('--seed', type=int, default=21, 
                        help='Random seed for initalization')
    parser.add_argument("--device", default="cuda", type=str, required=False,
                        help="cuda or cpu") 
    parser.add_argument('--num_classes', type=int, default=3, 
                        help='Number of classes to classify')
    parser.add_argument('--max_length', type=int, default=256, 
                        help='max sequence lenght')
    parser.add_argument('--stride', type=int, default=4, 
                        help='stride to cut sentence')
    
    parser.add_argument('--data_mode', type=str, default='single', 
                        help='single or pair sentence')
    parser.add_argument('--pool_type', type=str, default='sum', 
                        help='pooling type to represent term')


    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--get_conflict", action='store_true',
                        help="Consider conflict label")
    parser.add_argument("--print_model_info", action='store_true',
                        help="Print model information")
    parser.add_argument("--replace_term_with_mask", action='store_true',
                        help="replace term with mask token")
    parser.add_argument("--imb_weight", action='store_true',
                        help="Using imbalance weight")

    parser.add_argument("--mix_count", default=3, type=int,
                        help="Number of mix layers in mixlayers method"),
    parser.add_argument("--mix_type", default="HSUM", type=str, choices= ["HSUM", "PSUM"],
                        help="Type of mixation in mixlayers method"),

    parser.add_argument('--test_batch_size', type=int, default=32, 
                        help='(default=%(default)d)')
    parser.add_argument('--n_gpu', type=int, default=1, 
                        help='(default=%(default)d)')          
    args = parser.parse_args()
    return args


args = get_args()

args.label_weights = torch.Tensor(label_weights_acd_acsa).to(args.device)
if args.imb_weight:
    args.label_weights = label_weights_imb_acd_acsa

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# import required classes
if args.approach == "phobert":
    from trainer.ACD_ACSA_trainer import Trainer
    from model.ACD_ACSA_PhoBERT import Net
    from loader.load_acd_acsa import Loader, aspect_type_dict, aspect_sentiment_dict
    MODEL_CLASSES[args.approach] = (RobertaConfig, Net, PhobertTokenizer)
    keys_in_name = ["seed", "num_classes", "max_length", "lr", "epochs", "imb_weight", "data_mode", "pool_type", "weight_decay", "do_lower_case", "replace_term_with_mask", "train_batch_size", "test_batch_size"]
elif args.approach == "phobert_lstm":
    from trainer.ACD_ACSA_trainer import Trainer
    from model.ACD_ACSA_PhoBERT_LSTM import Net
    from loader.load_acd_acsa import Loader, aspect_type_dict, aspect_sentiment_dict
    MODEL_CLASSES[args.approach] = (RobertaConfig, Net, PhobertTokenizer)
elif args.approach == "phobert_mixlayer":
    from trainer.TBSA_trainer import Trainer
    from model.TBSA_PhoBERT_MixLayers import Net
    from loader.load_tbsa import Loader, aspect_type_dict, aspect_sentiment_dict
    MODEL_CLASSES[args.approach] = (RobertaConfig, Net, PhobertTokenizer)
    keys_in_name = ["seed", "num_classes", "max_length", "mix_type", "mix_count", "lr", "epochs", "imb_weight", "data_mode", "pool_type", "weight_decay", "do_lower_case", "replace_term_with_mask", "train_batch_size", "test_batch_size"]

args.aspect_type_dict = aspect_type_dict
args.aspect_sentiment_dict = aspect_sentiment_dict
if not args.get_conflict:
    args.label_weights = args.label_weights[:-1]
    args.aspect_sentiment_dict.pop('conflict', None)
args.num_classes = len(args.aspect_sentiment_dict)


def eval(model, data):
        total_loss = 0
        acc = 0
        steps = 0
        groundtruth_count = 0

        prediction = []
        groundtruths = []
        aspect_type_label = []

        for batch in tqdm(data, desc= "Evaluation"):
            model.eval()
            batch = [t.to(args.device) for t in batch]
            input_ids, attention_mask, token_type_ids, target_masks, aspect_type_labels, aspect_sentiment_labels = batch

            loss, logits = model(input_ids, attention_mask, target_masks, aspect_type_labels, aspect_sentiment_labels, 
                                      args.data_mode, args.pool_type, args.label_weights)

            out_prob = F.softmax(logits.view(-1, args.num_classes), 1)
            out_prob = out_prob.detach().cpu().numpy()
            label_ids = aspect_sentiment_labels.view(-1).to('cpu').numpy()
            preds = np.argmax(out_prob, axis=1)
            tmp_accuracy = (preds == label_ids)

            total_loss += loss.item()
            acc += np.sum(tmp_accuracy)

            steps += 1
            groundtruth_count += len(input_ids)

            prediction += list(preds)
            groundtruths += list(label_ids)
            aspect_type_label.extend(aspect_type_labels.detach().cpu().numpy())

        loss, acc = total_loss/ steps, acc/groundtruth_count
        micro, macro = mi_ma_cro_f1(groundtruths, prediction)
        micro_noNone, macro_noNone = mi_ma_cro_f1_noNone(groundtruths, prediction)
        confusion_matrix = get_confusion_matrix(groundtruths, prediction, args.aspect_sentiment_dict)
        strict_acc_acd = get_strict_acc_acd(groundtruths, prediction, args.aspect_type_dict, args.aspect_sentiment_dict)
        strict_acc_acd_acsa = get_strict_acc_acd_acsa(groundtruths, prediction, args.aspect_type_dict)
        return loss, acc, micro, macro, micro_noNone, macro_noNone, confusion_matrix, strict_acc_acd, strict_acc_acd_acsa, prediction, aspect_type_label

def main():
    set_seed(args)
    
    # Argument info
    print("\033[92mArguments:\033[0m")
    list_args = args._get_kwargs()
    for key, val in list_args:
        print("\t{}: {}".format(key,val))

    if "phobert" in args.approach.lower():
        model_files = MODEL_FILE["phobert"]
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.approach]
        if model_files['vocab_file'] is not None:
            tokenizer = tokenizer_class(vocab_file= model_files['vocab_file'], merges_file= model_files['merges_file'], do_lower_case= args.do_lower_case)
        else:
            tokenizer = tokenizer_class.from_pretrained(model_files['model_file'], do_lower_case= args.do_lower_case)
        tokenizer.do_lower_case = args.do_lower_case
    
    if args.approach == "phobert":
        config = config_class.from_pretrained(model_files['config_file'], num_labels= args.num_classes)
        model = model_class(model_files['model_file'], config= config, num_classes= args.num_classes, num_categories= len(args.aspect_type_dict))

    elif args.approach == "phobert_mixlayer":
        config = config_class.from_pretrained(model_files['config_file'], num_labels= args.num_classes, output_hidden_states= True)
        model = model_class(model_files['model_file'], config= config, num_classes= args.num_classes, count= args.mix_count, mix_type= args.mix_type)
    
    model.load_state_dict(torch.load(args.model_path, map_location= torch.device(args.device)))
    model.to(args.device)

    # Load predict data
    examples = get_examples(args.predict_file, get_conflict= args.get_conflict)
    dataset = convert_examples_to_dataset(examples, args.aspect_type_dict, args.aspect_sentiment_dict, args.max_length, tokenizer, args.replace_term_with_mask)
    Sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=Sampler, batch_size= args.test_batch_size)

    rs = eval(model, dataloader)
    print("Loss: {:.3f} || Accuracy: {:.3f} || macro_F1: {:.3f} || Accuracy_noNone: {:.3f} || macro_F1_noNone: {:.3f} || strict_acc_acd: {:.3f} || strict_acc_acd_acsa: {:.3f}".format(rs[0], rs[1]*100, rs[3]*100, rs[4]*100, rs[5]*100, rs[7]*100, rs[8]*100))
    print("Confusion matrix: \n{}".format(rs[6]))

    # Check wrong predictions
    reverse_aspect_type_dict = dict([(idx, label) for (label, idx) in args.aspect_type_dict.items()])
    reverse_aspect_sentiment_dict = dict([(idx, label) for (label, idx) in args.aspect_sentiment_dict.items()])

    aspect_type_labels = [reverse_aspect_type_dict[i] for i in rs[-1]]
    sentiment_predictions = [reverse_aspect_sentiment_dict[i] for i in rs[-2]]
    assert len(aspect_type_labels) == len(sentiment_predictions), "{} not equal to {}".format(len(aspect_type_labels), len(sentiment_predictions))

    num_aspect_type = len(args.aspect_type_dict)
    total_cases = int(len(sentiment_predictions)/num_aspect_type)
    assert total_cases == len(examples)

    for i in range(total_cases):
        for j in range(num_aspect_type):
            if sentiment_predictions[i*num_aspect_type + j] != "None":
                examples[i].predict_type.append(aspect_type_labels[i*num_aspect_type + j])
                examples[i].predict_sentiment.append(sentiment_predictions[i*num_aspect_type + j])

    max_cnt = 5
    for i, example in enumerate(examples):
        if i > max_cnt:
            break
        print(example.tokens)
        print(example.term)
        print(example.aspects)
        print(example.predict_type)
        print(example.predict_sentiment)
        print("=" * 100)
        

if __name__ == "__main__":
    main()
