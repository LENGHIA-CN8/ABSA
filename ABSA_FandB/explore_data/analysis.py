from explore_data.constant import attribute_dict, sentiment_dict
from explore_data.data_utils import extract_info_from_file, find_urls
import os
import json
import re
import argparse
from collections import Counter
# import matplotlib.pyplot as plt

def EOS_without_unclear(all_documents):
    doc_id = set()
    num_sentences = 0
    for doc in all_documents:
        for sent in doc.sentences:
            check_sent = False
            for term_id in sent.terms:
                term_text = [term.text for term in sent.terms[term_id]['term']]
                attributes = [attr.category for attr in sent.terms[term_id]['attribute']]
                if 'EOS' in term_text and len(attributes) == 0:
                    print("EOS has not been annotated:", doc.index)
                if 'EOS' in term_text and ('unclear' not in attributes or len(attributes) > 1):
                    doc_id.add(doc.index)
                    check_sent = True
            if check_sent:
                num_sentences += 1
    return doc_id, num_sentences

def EOS_with_unclear(all_documents):
    doc_id = set()
    num_sentences = 0
    for doc in all_documents:
        for sent in doc.sentences:
            check_sent = False
            for term_id in sent.terms:
                term_text = [term.text for term in sent.terms[term_id]['term']]
                attributes = [attr.category for attr in sent.terms[term_id]['attribute']]
                if 'EOS' in term_text and 'unclear' in attributes:
                    # assert len(term_text) == 1, f"{doc.index}"
                    # assert len(attributes) == 1, f"{doc.index}"
                    doc_id.add(doc.index)
                    check_sent = True
            if check_sent:
                num_sentences += 1
    return doc_id, num_sentences

def EOS_with_unclear_but_not_term(all_documents):
    doc_id = set()
    num_sentences = 0
    for doc in all_documents:
        for sent in doc.sentences:
            EOS_unclear = False
            sentimentss, attributess = [], []
            for term_id in sent.terms:
                term_text = [term.text for term in sent.terms[term_id]['term']]
                attributes = [attr.category for attr in sent.terms[term_id]['attribute']]
                
                if 'EOS' in term_text and 'unclear' in attributes:
                    EOS_unclear = True
                else:
                    sentimentss.extend([attr.sentiment for attr in sent.terms[term_id]['attribute']])
                    attributess.extend(attributes)

            sentimentss = set(sentimentss)
            attributess = set(attributess)

            if len(sentimentss) == 0 and len(attributess) == 0: # check only contain EOS- unclear
                continue

            if EOS_unclear and (sentimentss != set(["neutral"]) or attributess != set(["general"])):
                doc_id.add(doc.index)
                num_sentences += 1
                # print(sentimentss, attributess)
    return doc_id, num_sentences

def term_with_unclear(all_documents):
    doc_id = set()
    num_sentences = 0
    num_terms = 0
    for doc in all_documents:
        for sent in doc.sentences:
            check_sent = False
            for term_id in sent.terms:
                term_text = [term.text for term in sent.terms[term_id]['term']]
                attributes = [attr.category for attr in sent.terms[term_id]['attribute']]
                if 'EOS' not in term_text and 'unclear' in attributes:
                    assert len(attributes) == 1, f"{doc.index}: {term_text}"
                    doc_id.add(doc.index)
                    check_sent= True
                    num_terms += 1
            if check_sent:
                num_sentences += 1
    return doc_id, num_sentences, num_terms

def term_with_no_attribute(all_documents):
    doc_id = list()
    for doc in all_documents:
        for sent_id, sent in enumerate(doc.sentences):
            EOS_unclear_in_sent = False # only consider the sentences that do not contain EOS - unclear
            for term_id in sent.terms:
                term_text = [term.text for term in sent.terms[term_id]['term']]
                attributes = [attr.category for attr in sent.terms[term_id]['attribute']]
            #     if 'EOS' in term_text and 'unclear' in attributes:
            #         EOS_unclear_in_sent = True
            #         break
            
            # if EOS_unclear_in_sent:
            #     continue

            for term_id in sent.terms:
                term_text = [term.text for term in sent.terms[term_id]['term']]
                attributes = [attr.category for attr in sent.terms[term_id]['attribute']]
                if 'EOS' not in term_text and len(attributes) == 0 and len(term_text) > 0:
                    doc_id.append(doc.index)
                    print(doc.index, sent_id)
                    # print(doc.index, term_id, term_text[0])
                    # sent.info()
    doc_id = Counter(doc_id)
    return sorted(doc_id.items(), key= lambda x: x[0])

def unannotated(all_documents):
    doc_id = set()
    num_sentences = 0
    for doc in all_documents:
        for sent in doc.sentences:
            if len(sent.terms) == 0:
                doc_id.add(doc.index)
                num_sentences += 1
    return doc_id, num_sentences

def check_missing_character(all_documents):
    # def _is_whitespace(c):
    #     if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    #         return True
    #     return False

    doc_id = set()
    for doc in all_documents:
        for sent in doc.sentences:
            sent_text = sent.text
            for term_id in sent.terms:
                for term in sent.terms[term_id]['term']:
                    if term.text == "EOS":
                        continue
                    assert term.text == sent_text[term.start_pos: term.end_pos], f"{term.text} and {sent_text[term.start_pos: term.end_pos]}"
                    if term.start_pos != 0:
                        if term.end_pos == len(sent_text):
                            if sent_text[term.start_pos - 1].isalpha():
                                doc_id.add(doc.index)
                                print(doc.index, term.text)
                        else:
                            if sent_text[term.start_pos - 1].isalpha() or sent_text[term.end_pos].isalpha():
                                doc_id.add(doc.index)
                                print(doc.index, term.text)
                    else:
                        if sent_text[term.end_pos].isalpha():
                            doc_id.add(doc.index)
                            print(doc.index, term.text)
                    
    return doc_id

def check_term_in_link(all_documents):
    doc_id = set()
    for doc in all_documents:
        for sent in doc.sentences:
            sent_text = sent.text
            urls = find_urls(sent_text)

            # if len(urls) > 0:
            #     print(urls)
            urls_span = []
            for url in urls:
                start_url = sent_text.index(url)
                end_url = start_url + len(url)
                urls_span.append((start_url, end_url))

            for term_id in sent.terms:
                for term in sent.terms[term_id]['term']:
                    if term.text == "EOS":
                        continue
                    
                    assert term.text == sent_text[term.start_pos: term.end_pos], f"{term.text} and {sent_text[term.start_pos: term.end_pos]}"
                    for start_url, end_url in urls_span:
                        if term.start_pos >= start_url and term.start_pos < end_url:
                            doc_id.add(doc.index)
                            print(doc.index, term.text)
    return doc_id

def check_mistake_between_EOSandterm(all_documents):
    doc_id = set()
    for doc in all_documents:
        for sent in doc.sentences:
            for term_id in sent.terms:
                  term_text = [term.text for term in sent.terms[term_id]['term']]
                  if 'EOS' in term_text:
                      for term in sent.terms[term_id]['term']:
                          if term.end_pos != len(sent.text.rstrip()):
                              doc_id.add(doc.index)
                              print(doc.index, sent.text, term.end_pos, len(sent.text.rstrip()))
                  else:
                      for term in sent.terms[term_id]['term']:
                          if len(sent.text) - term.start_pos < 3:
                              doc_id.add(doc.index)
                              print(doc.index, sent.text, term.end_pos, len(sent.text))
    return doc_id


def get_args():
    parser = argparse.ArgumentParser(description='TBSA')
    parser.add_argument('--data_dir', type=str, 
                        help='data_dir')
    args = parser.parse_args()
    return args

def main(args):
    data_root = args.data_dir
    all_documents = []
    for folder in os.listdir(data_root):
        # if len(folder) > 4:
        #     continue
        # if int(folder[2]) >= 2:
        #     continue
        for file in os.listdir(os.path.join(data_root, folder)):
            if ".ann" in file:
                try:
                    all_documents.append(extract_info_from_file(os.path.join(data_root, folder, file)))
                except Exception as e:
                    print(file, "\n", e)

    attributes_stat = {}
    number_of_sentences = 0
    number_of_distant_terms = 0
    number_of_all_terms = 0
    for attribute in attribute_dict:
        attributes_stat[attribute] = {sentiment: 0 for sentiment in sentiment_dict}
    for doc in all_documents:
        number_of_sentences += doc.get_num_sent()
        number_of_distant_terms += doc.get_num_terms()[0]
        number_of_all_terms += doc.get_num_terms()[1]
        doc_stas = doc.statistic_attributes(attribute_dict, sentiment_dict)
        for attribute in attribute_dict:
            for sentiment in sentiment_dict:
                attributes_stat[attribute][sentiment] += doc_stas[attribute][sentiment]

    print("==== Statistic ====")
    print("Number of sentences:", number_of_sentences)
    print("Number of terms:", number_of_distant_terms, int(number_of_all_terms), "(also contains EOS)")
    print(json.dumps(attributes_stat, indent= 3))
    print()
    print("==== Check =====")

    num_EOS_without_unclear = EOS_without_unclear(all_documents)
    print("- EOS without unclear:", len(num_EOS_without_unclear[0]), num_EOS_without_unclear[1], num_EOS_without_unclear)
    num_EOS_with_unclear = EOS_with_unclear(all_documents)
    print("- EOS with unclear:", len(num_EOS_with_unclear[0]), num_EOS_with_unclear[1], num_EOS_with_unclear[0])
    num_EOS_with_unclear_but_not_term =  EOS_with_unclear_but_not_term(all_documents)
    print("- EOS with unclear but term has meaning:", len(num_EOS_with_unclear_but_not_term[0]), num_EOS_with_unclear_but_not_term[1])
    num_term_with_unclear = term_with_unclear(all_documents)
    print("- Terms with unclear:", len(num_term_with_unclear[0]), num_term_with_unclear[1], num_term_with_unclear[2], num_term_with_unclear[0])
    num_unannotated = unannotated(all_documents)
    print("Unannotated:", len(num_unannotated[0]), num_unannotated[1])
    print() 
    print("==== ERROR ====")
    print("- Terms with no attribute:", term_with_no_attribute(all_documents))
    print("- Terms in link:", check_term_in_link(all_documents))
    print("- Maybe wrong annotate term and EOS:", check_mistake_between_EOSandterm(all_documents))
    print("- Maybe term missing character:", check_missing_character(all_documents))


    # Write EOS unclear and unannotated doc
    with open(data_root.replace("raw", "unannotated.txt"), "w") as f:
        for line in sorted(list(num_unannotated[0])):
            f.write(line + "\n")


if __name__ == "__main__":
    args = get_args()
    main()