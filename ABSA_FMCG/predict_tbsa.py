from constant import label_weights_tbsa, label_weights_imb_tbsa, MODEL_FILE
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from time import time
from utils import print_model_report
from transformers import PhobertTokenizer, RobertaConfig
import torch, random
import numpy as np
import os
import argparse
from loader.load_tbsa import get_examples, convert_examples_to_dataset
from evaluate.evaluate_tbsa import mi_ma_cro_f1, get_confusion_matrix
from tqdm import tqdm

MODEL_CLASSES = {
}

def get_args():
    parser = argparse.ArgumentParser(description='TBSA')
    # Arguments
    parser.add_argument('--task', type=str, choices= ["TBSA"], default='TBSA', 
                        help='TBSA')
    parser.add_argument('--approach', type=str, default='phobert', 
                        help='Model type')
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="predict_file")  
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="path of trained model")  
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
    parser.add_argument('--pool_type', type=str, default='avg', 
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

args.label_weights = torch.Tensor(label_weights_tbsa).to(args.device)
if args.imb_weight:
    args.label_weights = label_weights_imb_tbsa

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 1:
        torch.cuda.manual_seed_all(args.seed)

# import required classes
if args.approach == "ABSA1":
    from trainer.Lstm_trainer import Trainer
    from model.ABSA1 import Net
    from loader.LSTM_loader import Loader
elif args.approach == "ABSA2":
    from trainer.Lstm_trainer import Trainer
    from model.ABSA2 import Net
    from loader.LSTM_loader import Loader
elif args.approach == "phobert":
    from trainer.TBSA_trainer import Trainer
    from model.TBSA_PhoBERT import Net
    from loader.load_tbsa import Loader, label_dict
    MODEL_CLASSES[args.approach] = (RobertaConfig, Net, PhobertTokenizer)
    keys_in_name = ["seed", "num_classes", "max_length", "lr", "epochs", "imb_weight", "data_mode", "pool_type", "weight_decay", "do_lower_case", "replace_term_with_mask", "train_batch_size", "test_batch_size"]
elif args.approach == "phobert_lstm":
    from trainer.TBSA_trainer import Trainer
    from model.TBSA_PhoBERT_LSTM import Net
    from loader.load_tbsa import Loader, label_dict
    MODEL_CLASSES[args.approach] = (RobertaConfig, Net, PhobertTokenizer)
elif args.approach == "phobert_mixlayer":
    from trainer.TBSA_trainer import Trainer
    from model.TBSA_PhoBERT_MixLayers import Net
    from loader.load_tbsa import Loader, label_dict
    MODEL_CLASSES[args.approach] = (RobertaConfig, Net, PhobertTokenizer)
    keys_in_name = ["seed", "num_classes", "max_length", "mix_type", "mix_count", "lr", "epochs", "imb_weight", "data_mode", "pool_type", "weight_decay", "do_lower_case", "replace_term_with_mask", "train_batch_size", "test_batch_size"]

args.label_dict = label_dict
if not args.get_conflict:
    args.label_weights = args.label_weights[:-1]
    args.label_dict.pop('conflict', None)
args.num_classes = len(args.label_weights)

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
        model = model_class(model_files['model_file'], config= config, num_classes= args.num_classes)

    elif args.approach == "phobert_mixlayer":
        config = config_class.from_pretrained(model_files['config_file'], num_labels= args.num_classes, output_hidden_states= True)
        model = model_class(model_files['model_file'], config= config, num_classes= args.num_classes, count= args.mix_count, mix_type= args.mix_type)

    model.load_state_dict(torch.load(args.model_path, map_location= torch.device(args.device)))
    model.to(args.device)

    examples = get_examples(args.predict_file, get_conflict= args.get_conflict)
    dataset = convert_examples_to_dataset(examples, args.label_dict, args.max_length, tokenizer, args.replace_term_with_mask)
    Sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=Sampler, batch_size= args.test_batch_size)

    def eval(model, data):
        total_loss = 0
        acc = 0
        steps = 0
        groundtruth_count = 0

        prediction = []
        groundtruths = []
        bin_masks = []

        for batch in tqdm(data, desc= "Evaluation"):
            model.eval()
            batch = [t.to(args.device) for t in batch]
            input_ids, attention_mask, token_type_ids, target_masks, labels = batch
            loss, logits = model(input_ids, attention_mask, target_masks, labels, args.data_mode, args.pool_type, torch.Tensor(args.label_weights).to(args.device))

            out_prob = F.softmax(logits.view(-1, args.num_classes), 1)
            out_prob = out_prob.detach().cpu().numpy()
            label_ids = labels.view(-1).to('cpu').numpy()
            preds = np.argmax(out_prob, axis=1)
            tmp_accuracy = (preds == label_ids)

            total_loss += loss.item()
            acc += np.sum(tmp_accuracy)

            steps += 1
            groundtruth_count += len(input_ids)

            prediction += list(preds)
            groundtruths += list(label_ids)

        loss, acc = total_loss/ steps, acc/groundtruth_count
        micro, macro = mi_ma_cro_f1(groundtruths, prediction)
        confusion_matrix = get_confusion_matrix(groundtruths, prediction, args.label_dict)
        return loss, acc, micro, macro, confusion_matrix, prediction

    rs = eval(model, dataloader)
    print("Loss: {:.3f} || Accuracy: {:.3f} || macro_F1: {:.3f} ||".format(rs[0], rs[1]*100, rs[3]*100))
    print("Confusion matrix: \n{}".format(rs[4]))

    reverse_label_dict = dict([(idx, label) for (label, idx) in args.label_dict.items()])
    prediction = [reverse_label_dict[i] for i in rs[-1]]

    for i, example in enumerate(examples):
        example.predict = prediction[i]

    max_cnt = 0
    for example in examples:
        if example.label != example.predict:
        # if example.label == "negative" and example.predict == "neutral":
            max_cnt += 1
            print(example.tokens)
            print(example.term)
            print(example.predict, "||", example.label)
            print("="*100)
        
        if max_cnt > 5:
            break
    # print(max_cnt)
if __name__ == "__main__":
    main()




