import os
import numpy as np
import collections

class Document:
    def __init__(self, index):
        self.index = index
        self.sentences = list()

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def get_num_sent(self):
        return(len(self.sentences))
    
    def get_num_terms(self):
        distinct_terms, all_terms = 0, 0
        for sent in self.sentences:
            sent_terms = sent.get_num_terms()
            distinct_terms += sent_terms[0]
            all_terms += sent_terms[1]
        return distinct_terms, all_terms

    def statistic_attributes(self, attribute_dict, sentiment_dict):
        attributes_stat = {}
        for attribute in attribute_dict:
            attributes_stat[attribute] = {sentiment: 0 for sentiment in sentiment_dict}
        for sentence in self.sentences:
            sent_stas = sentence.statistic_attributes(attribute_dict, sentiment_dict)
            for attribute in attribute_dict:
                for sentiment in sentiment_dict:
                    attributes_stat[attribute][sentiment] += sent_stas[attribute][sentiment]
        return attributes_stat

class Sentence:
    def __init__(self, text, start_pos, end_pos):
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.terms = {}
    

    def add_term(self, orig_idx, term):
        if orig_idx not in self.terms:
            self.terms[orig_idx] = {'term': list(), 'attribute': list()}   
        self.terms[orig_idx]['term'].append(term)

    def add_attribute(self, term_idx, attribute):
        self.terms[term_idx]['attribute'].append(attribute)

    def get_num_terms(self):
        distinct_terms = len(self.terms)
        all_terms = np.sum([len(self.terms[term_id]['term']) for term_id in self.terms]) #contain EOS
        return distinct_terms, all_terms

    def statistic_attributes(self, attribute_dict, sentiment_dict):
        attributes_stat = {}
        for attribute in attribute_dict:
            attributes_stat[attribute] = {sentiment: 0 for sentiment in sentiment_dict}
        for id in self.terms:
            for attr_senti in self.terms[id]['attribute']:
                attributes_stat[attr_senti.category][attr_senti.sentiment] += 1
        return attributes_stat

    def info(self):
        print("Text:", self.text)
        print("start: {}, end: {}".format(self.start_pos, self.end_pos))
        print("---- All terms {} ----".format(self.get_num_terms()))
        for term_idx in self.terms:
            print("ID:", term_idx)
            for term in self.terms[term_idx]['term']:
                print("term: {} || from: {} || to: {}".format(term.text, term.start_pos, term.end_pos))
            for attribute in self.terms[term_idx]['attribute']:
                print("category: {} || sentiment: {}".format(attribute.category, attribute.sentiment))
            print("-"*100)

class Term:
    def __init__(self, text, start_pos, end_pos):
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def __len__(self):
        return len(self.text)

class Attribute:
    def __init__(self, category, sentiment):
        self.category = category
        self.sentiment = sentiment
