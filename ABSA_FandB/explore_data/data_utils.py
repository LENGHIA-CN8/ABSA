from explore_data.data_instance import *
import os
import re
import json
import emoji
from underthesea import word_tokenize
from constant import *

def extract_info_from_file(filename: str):
    """
    :param:
        - filename: .ann file
    :return:
        - doc: Document
    """
    with open(filename, 'r', encoding= 'utf-8') as f:
        ann_lines = [line.replace("\n", "") for line in f.readlines()]
    raw = filename.replace(".ann",".txt")
    with open(raw, "r", encoding= 'utf-8') as f:
        raw_lines = [line.replace("\n", "") for line in f.readlines()]

    map_2_org_idx = {} # map coref to original index
    for line in ann_lines:
        if line[0] == "R":
            _, _, arg1, arg2 = line.split()
            map_2_org_idx[arg1.split(":")[-1]] = arg2.split(":")[-1]

    doc = Document(filename.split("/")[-1].replace(".ann", ""))
    
    accumlated_len = 0
    for sent in raw_lines:
        text = sent
        start_pos = accumlated_len
        accumlated_len += len(sent) + 1 # plus 1 to contain "\n" character has been removed in line 17
        end_pos = accumlated_len

        sentence = Sentence(text, start_pos, end_pos)
        doc.add_sentence(sentence)

    map_term_to_sent = {} # map term to belonged sentence index of document
    for line in ann_lines:
        if 'T' == line[0]: # get term and map to belong sentence
            term_idx, info, term_text = line.split("\t")
            type, start, end = info.split()
            start_term, end_term = int(start), int(end)
            
            term_idx = map_2_org_idx[term_idx] if term_idx in map_2_org_idx else term_idx

            for sent_id, sent in enumerate(doc.sentences):
                start_sent, end_sent = sent.start_pos, sent.end_pos
                if start_term >= start_sent and start_term < end_sent:
                    new_start_term, new_end_term = start_term-start_sent, end_term-start_sent
                    assert term_text == sent.text[new_start_term:new_end_term], f"{term_text}, {sent.text[new_start_term:new_end_term]}"
                    
                    map_term_to_sent[term_idx] = sent_id

                    term_text = term_text if type != "EOS" else "EOS"
                    term = Term(term_text, new_start_term, new_end_term)
                    sent.add_term(term_idx, term)
            
        if 'A' == line[0]: # get attribute for term
            line_info = line.split()

            if len(line_info) == 4:
                attr_idx, category, term_idx, sentiment = line.split()
            else:
                attr_idx, category, term_idx = line.split()
                sentiment = 'None'

            attribute = Attribute(category, sentiment)
            doc.sentences[map_term_to_sent[term_idx]].add_attribute(term_idx, attribute)

    return doc

def find_urls(text):
    # urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    # urls += re.findall('www?.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    urls = re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+', text)
    filter_urls = set()
    for url in urls:
        if url[url.index(".")+1].isdigit() and url[url.index(".")-1].isdigit():
            continue
        filter_urls.add(url)
    return filter_urls

def remove_urls(text):
    urls = find_urls(text)  
    for url in urls:
        text = text.replace(url, " ")
    return text

def remove_emoji(text):
    new_text = re.sub(emoji.get_emoji_regexp(), " ", text)
    return new_text

def remove_special_character(text):
    new_text = re.sub(r'[|\\!()\"\':;()\[\]{}#%^&*\-+=`~></“”]', " ", text)
    return new_text

def word_segment(text):
    if len(text.split(" ")) == 1:
        return text
    words = [word.strip().replace(" ", "_") for word in word_tokenize(text)]
    return " ".join(words)

def preprocess(sentence: Sentence, rm_url = True, rm_emoji= True, rm_spec_ch= True, wseg = True):
    """
    This function can use to remove all urls, emoji, special characters
    and word segmentation. Sentence is splitted into pieces (depend on 
    terms position). Urls, emoji, characters are removed in each piece.
    """
    if not rm_url and not rm_emoji and not rm_spec_ch and not wseg:
        return sentence
    
    sent_text = sentence.text

    # extract all term excepts EOS and re-order by start position
    all_terms_except_EOS = []
    for term_id in sentence.terms:
        for term in sentence.terms[term_id]['term']:
            if term.text == "EOS":
                continue
            all_terms_except_EOS.append((term_id, term))
    all_terms_except_EOS = sorted(all_terms_except_EOS, key= lambda x: x[1].start_pos)
    
    # extract pieces
    pieces = []
    cur_length = 0
    for i, (term_id, term) in enumerate(all_terms_except_EOS):
        sentence.terms[term_id]['term'] = [] # reset to none list
        assert term.text == sent_text[term.start_pos: term.end_pos], f"{term.text} vs {sent_text[term.start_pos: term.end_pos]}"
        if i == len(all_terms_except_EOS) - 1:
            pieces.append(sent_text[cur_length:term.start_pos])
            pieces.append(term.text)
            pieces.append(sent_text[term.end_pos:])
        else:    
            pieces.append(sent_text[cur_length:term.start_pos])
            pieces.append(term.text)
            cur_length = term.end_pos

    if len(all_terms_except_EOS) > 0: # sentence does not contain only EOS
        assert "".join(pieces) == sent_text, "{} \n\n {}".format("".join(pieces), sent_text)

    # preprocess in each piece and concat to new_sentence
    new_sentence = ""
    for i, (term_id, term) in enumerate(all_terms_except_EOS):
        piece_clean = pieces[2*i]
        
        term.text = term.text.strip().replace(" ", "_") # term text is always joined to 1 word
        if wseg:
            piece_clean = word_segment(piece_clean)

        if rm_url:
            piece_urls = find_urls(pieces[2*i])  
            for url in piece_urls:
                piece_clean = piece_clean.replace(url, " ")

        if rm_emoji:
            piece_clean = remove_emoji(piece_clean)

        if rm_spec_ch:
            piece_clean = remove_special_character(piece_clean)

        piece_clean = " ".join(piece_clean.split()).strip()

        new_sentence += piece_clean + " "
        # map term to new position
        term.start_pos = len(new_sentence)
        term.end_pos = term.start_pos + len(term.text)

        sentence.terms[term_id]['term'].append(term) 
        new_sentence += term.text + " "

        if i == len(all_terms_except_EOS) - 1: #process for the last piece
            piece_clean = pieces[-1]
            if wseg:
                piece_clean = word_segment(piece_clean)

            if rm_url:
                piece_urls = find_urls(pieces[-1])
                for url in piece_urls:
                    piece_clean = piece_clean.replace(url, "")
            
            if rm_emoji:
                piece_clean = remove_emoji(piece_clean)
            
            if rm_spec_ch:
                piece_clean = remove_special_character(piece_clean)

            piece_clean = " ".join(piece_clean.split()).strip()
            new_sentence += piece_clean

    if len(new_sentence) == 0: # sentence contains only EOS 
        if wseg:
            new_sentence = word_segment(sent_text)

        if rm_url:
            piece_urls = find_urls(new_sentence)  
            for url in piece_urls:
                new_sentence = new_sentence.replace(url, " ")

        if rm_emoji:
            new_sentence = remove_emoji(new_sentence)

        if rm_spec_ch:
            new_sentence = remove_special_character(new_sentence)
    assert len(new_sentence) > 0

    # re-check and map EOS token
    term_count = 0
    sentence.text = new_sentence
    for term_id in sentence.terms:
        for term in sentence.terms[term_id]['term']:
            if term.text == 'EOS': # map for EOS token
                term.start_pos = len(sentence.text) - 1
                term.end_pos = len(sentence.text)
            else:
                term_count += 1
                assert sentence.text[term.start_pos: term.end_pos] == term.text, "{} vs {}".format(term.text, sentence.text[term.start_pos: term.end_pos])
    assert term_count == len(all_terms_except_EOS), "{} vs {}".format(term_count, len(all_terms_except_EOS))
    return sentence
            

def check_unrelated_sentence(sentence, keep_EOS_unclear_but_not_term= False):
    """
    keep_EOS_unclear_but_not_term: True - keep the sentence that contain EOS - unclear
    but the term is not only general - neutral. Else all sentences contain EOS - unclear
    is considered as unrelated.
    """
    EOS_unclear = False
    term_with_meaning = False
    sentimentss, attributess = [], []
    
    for term_id in sentence.terms:
        terms = sentence.terms[term_id]['term']
        attributes = sentence.terms[term_id]['attribute']

        if "EOS" == terms[0].text and "unclear" == attributes[0].category:
            EOS_unclear = True
        else:    
            sentimentss.extend([attr.sentiment for attr in attributes])
            attributess.extend([attr.category for attr in attributes])

    sentimentss = set(sentimentss)
    attributess = set(attributess)
    
    if len(sentimentss) == 0 and len(attributess) == 0:
        return True
    
    term_with_meaning = (sentimentss != set(['neutral']) or attributess != set(["general"]))

    if not keep_EOS_unclear_but_not_term and EOS_unclear:
        return True   
    elif keep_EOS_unclear_but_not_term and EOS_unclear and not term_with_meaning:
        return True

    return False

if __name__ == "__main__":
    doc = extract_info_from_file("../data/FMCG/raw/no18/f185.ann")
    print(doc.index)
    print(doc.get_num_sent())
    print(json.dumps(doc.statistic_attributes(attribute_dict, sentiment_dict), indent= 4))
    for sent in doc.sentences:
        if len(sent.terms) == 0:
            continue
        sent.info()
        sent = preprocess(sent, rm_url = True, rm_emoji= True, rm_spec_ch= True, wseg= True)
        print("\n\n")
        sent.info()
        print("="*100)
    
