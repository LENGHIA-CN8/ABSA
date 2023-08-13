from constant import label_weights_tbsa, label_weights_imb_tbsa, MODEL_FILE
from time import time
from utils import print_model_report
from transformers import PhobertTokenizer, RobertaConfig
import torch, random
import numpy as np
import os
import argparse

MODEL_CLASSES = {
}

def get_args():
    parser = argparse.ArgumentParser(description='TBSA')
    # Arguments
    parser.add_argument('--task', type=str, choices= ["TBSA"], default='TBSA', 
                        help='TBSA')
    parser.add_argument('--approach', type=str, default='phobert', 
                        help='Model type')
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="Data folder contains train, dev, test")  
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
    parser.add_argument('--epochs', type=int, default=4, 
                        help='Number of epochs')
    
    parser.add_argument('--data_mode', type=str, default='single', 
                        help='single or pair sentence')
    parser.add_argument('--pool_type', type=str, default='avg', 
                        help='pooling type to represent term')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help='accumulated gradient')
    parser.add_argument('--optim', type=str, default="AdamW", 
                        help='Optimization method')
    parser.add_argument('--lr', type=float, default=2e-5, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, 
                        help='weight_decay for optimizer')
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
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

    parser.add_argument('--train_batch_size', type=int, default=32, 
                        help='(default=%(default)d)')
    parser.add_argument('--test_batch_size', type=int, default=32, 
                        help='(default=%(default)d)')
    parser.add_argument('--n_gpu', type=int, default=1, 
                        help='(default=%(default)d)')
    args = parser.parse_args()
    return args


args = get_args()
if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                         args.gradient_accumulation_steps))

args.label_weights = label_weights_tbsa
if args.imb_weight:
    args.label_weights = label_weights_imb_tbsa
    
args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
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
    keys_in_name = ["seed", "num_classes", "max_length", "stride", "lr", "epochs", "imb_weight", "data_mode", "pool_type", "weight_decay", "do_lower_case", "replace_term_with_mask", "train_batch_size"]
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
    keys_in_name = ["seed", "num_classes", "max_length", "stride", "mix_type", "mix_count", "lr", "epochs", "imb_weight", "data_mode", "pool_type", "weight_decay", "do_lower_case", "replace_term_with_mask", "train_batch_size"]

args.label_dict = label_dict
if not args.get_conflict:
    args.label_weights = args.label_weights[:-1]
    args.label_dict.pop('conflict', None)
args.num_classes = len(args.label_weights)

def main():
    set_seed(args)

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
        
    # Load data
    print("\033[92mLoading data ...\033[0m")
    DataLoader = Loader(args, label_dict, tokenizer)
    train_dataloader, valid_dataloader, test_dataloader = DataLoader.load_data()

    # Argument info
    print("\033[92mArguments:\033[0m")
    list_args = args._get_kwargs()
    for key, val in list_args:
        print("\t{}: {}".format(key,val))

    # Log
    if not os.path.isdir("../log/FMCG/"+ args.task):
        os.mkdir("../log/FMCG/"+ args.task)
    log_file = "../log/FMCG/" + args.task + "/" + args.approach
    for key, val in list_args:
        if key not in keys_in_name:
            continue
        log_file += "_{}_{}".format(key,str(val))
    log_file = log_file + ".txt"
    print("\tLog output: {}".format(log_file))
    f_log = open(log_file, "w")

    if args.print_model_info:
        print_model_report(model) # model info

    # Trainer
    trainer = Trainer(args, model, train_dataloader, valid_dataloader, test_dataloader, f_log, device= args.device)

    print("\033[92mTraining ...\033[0m")
    time1 = time()
    # torch.autograd.set_detect_anomaly(True)
    trainer.train()
    time2 = time()
    print("\nEstimate time: {:.3f}h".format((time2 - time1)/3600))
    print("\nEstimate time: {:.3f}h".format((time2 - time1)/3600), file= f_log)

    print("\033[92mTesting ...\033[0m")
    time_test1 = time()
    test_loss, test_acc, test_micro_f1, test_macro_f1, test_confusion = trainer.eval(trainer.model, trainer.test_loader)
    time_test2 = time()

    print("Confusion matrix")
    print("Confusion matrix\n{}".format(test_confusion), file= f_log)
    print(test_confusion)

    print("Test: loss= {:.3f}, acc= {:.3f}%, mirco_f1= {:.3f}%, macro_f1= {:.3f}%".format(test_loss, test_acc*100, test_micro_f1*100, test_macro_f1*100))
    print("Infer time: {:.3f}ms".format(1000* (time_test2- time_test1)/ (args.test_batch_size*len(test_dataloader))))
    print("Test: loss= {:.3f}, acc= {:.3f}%, mirco_f1= {:.3f}%, macro_f1= {:.3f}%".format(test_loss, test_acc*100, test_micro_f1*100, test_macro_f1*100), file= f_log)
    print("Infer time: {:.3f}ms".format(1000* (time_test2- time_test1)/ (args.test_batch_size*len(test_dataloader))), file= f_log)
    f_log.close()

    # Save model
    if not os.path.isdir("../saved_model/FMCG/" + args.task):
        os.mkdir("../saved_model/FMCG/" + args.task)
    save_model_path = log_file.replace("log", "saved_model").replace(".txt", ".pt")
    torch.save(trainer.model.state_dict(), save_model_path)

    # Save result
    import csv
    if not os.path.isdir("../result/FMCG/" + args.task):
        os.mkdir("../result/FMCG/" + args.task)
        
    save_result_path = os.path.join("../result/FMCG/", args.task, args.approach + ".csv")

    print("Saving log to", log_file)
    print("Saving model to", save_model_path)
    print("Saving result to", save_result_path)
    if not os.path.isfile(save_result_path):
        with open(save_result_path, mode='a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([key for key, val in list_args if key in keys_in_name] + ["acc", "micro_f1", "macro_f1", "valid_macro_f1"])
            writer.writerow([val for key, val in list_args if key in keys_in_name] + ["{:.3f}".format(test_acc*100), "{:.3f}".format(test_micro_f1*100), "{:.3f}".format(test_macro_f1*100), "{:.3f}".format(trainer.best_valid_macro_f1*100)])
    else:
        with open(save_result_path, mode='a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([val for key, val in list_args if key in keys_in_name] + ["{:.3f}".format(test_acc*100), "{:.3f}".format(test_micro_f1*100), "{:.3f}".format(test_macro_f1*100), "{:.3f}".format(trainer.best_valid_macro_f1*100)])
    
if __name__ == "__main__":
    main()