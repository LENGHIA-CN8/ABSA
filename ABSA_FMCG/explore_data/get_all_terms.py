import os
import argparse

def extract_terms_from_file(filename, not_term):
    terms = []
    with open(filename, "r") as f:
        for line in f.readlines():
            if "EOS" in line:
                continue
            line = line.replace("\n", "")
            if 'T' == line[0]:
                term_idx, info, term = line.split("\t")

                term = term.lower()
                if term == "epsi":
                    print(filename)
                if term not in not_term and not term.isdigit():
                    terms.append(term)
                else:
                    print(filename, term)
                    pass
    return terms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get all terms')
    parser.add_argument('--data_dir', type=str, 
                        help='data_dir')
    parser.add_argument('--domain', type=str, choices= ["FMCG", "FandB"],
                        help='data domain')
    args = parser.parse_args()
    data_root = args.data_dir

    all_terms = []
    not_terms = []
    for subfolder in os.listdir(data_root):
        for file in os.listdir(os.path.join(data_root, subfolder)):
            if ".ann" in file:
                all_terms.extend(extract_terms_from_file(os.path.join(data_root, subfolder, file), not_terms))

    all_terms = set(all_terms)
    with open("resource/term_vocab_" + args.domain + ".txt", "w") as f:
        for term in all_terms:
            f.write(term + "\n")

    