from data_utils import preprocess, extract_info_from_file, check_unrelated_sentence
import os
import json
import argparse

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def get_args():
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--raw_folder', type=str, required= True, help='Raw data folder')
    parser.add_argument('--save_file', type=str, required= True, help='Save file')
    parser.add_argument('--max_term_per_sent', type=int, help='Remove sentence that contains more than *max_term_per_sent terms')
    parser.add_argument('--wseg', action='store_true', help='Do word segmentation in preprocess')
    parser.add_argument('--rm_eos_unclear', action='store_true', help='Remove sentence that contains EOS-unclear annotation')
    args = parser.parse_args()
    return args



def main(args):
    all_data = {}
    all_documents = []
    for folder in os.listdir(args.raw_folder):
        for file in os.listdir(os.path.join(args.raw_folder,folder)):
            if ".ann" in file:
                all_documents.append(extract_info_from_file(os.path.join(args.raw_folder, folder, file)))

    raw_num_distince_terms = 0
    raw_num_sentences = 0

    clean_num_distince_terms = 0
    clean_num_sentences = 0
    unrelated_and_unannotated = 0
    too_many_terms = 0
    for doc in all_documents:
        all_data[doc.index] = []      
        new_sentences = []
        
        # clean data
        raw_num_sentences += len(doc.sentences)
        for sentence in doc.sentences:
            raw_num_distince_terms  += len(sentence.terms)
            
            # We don't want to contain sentences that are unrelated to domain 
            # (contain EOS - unclear - optional) or haven't been annotated
            if args.rm_eos_unclear:
                if check_unrelated_sentence(sentence, False): # contain EOS - unclear
                    unrelated_and_unannotated += 1
                    continue

            if len(sentence.terms) == 0: # haven't been annotated
                unrelated_and_unannotated += 1
                continue

            if len(sentence.terms) > args.max_term_per_sent:
                too_many_terms += 1
                continue

            new_sentence = preprocess(sentence, wseg= args.wseg) # word segmentation, remove url, emoji, special characters
            new_sentences.append(new_sentence)

        clean_num_sentences += len(new_sentences)
        # reformat data text to words, and dictionary type
        for sentence in new_sentences:
            clean_num_distince_terms += len(sentence.terms)
            
            new_sentence = {}
            
            doc_tokens = []
            char_to_word_offset = [] # map character position to token position
            prev_is_whitespace = True

            # Split on whitespace so that different tokens may be attributed to their original position.
            for c in sentence.text:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(int(len(doc_tokens) - 1))

            new_sentence['tokens'] = doc_tokens
            new_sentence['terms'] = {}
            for term_id in sentence.terms:
                if "EOS" == sentence.terms[term_id]['term'][0].text and "unclear" == sentence.terms[term_id]['attribute'][0].category: # remove term EOS unclear
                    continue
                
                new_sentence['terms'][term_id] = {}
                new_sentence['terms'][term_id]['term'] = []

                for term in sentence.terms[term_id]['term']:
                    term_text = term.text
                    term_position = char_to_word_offset[term.start_pos]
                    
                    assert char_to_word_offset[term.start_pos] == char_to_word_offset[term.end_pos - 1], "{}: {} vs {}".format(term.text, char_to_word_offset[term.start_pos], char_to_word_offset[term.end_pos - 1])
                    if term.text == "EOS":
                        assert term_position == len(new_sentence['tokens']) - 1
                    else:
                        assert term_text == new_sentence['tokens'][term_position]
                    new_sentence['terms'][term_id]['term'].append((term_text, term_position))
    
                new_sentence['terms'][term_id]['attribute'] = []
                for attribute in sentence.terms[term_id]['attribute']:
                    new_sentence['terms'][term_id]['attribute'].append((attribute.category, attribute.sentiment))
            
            if len(new_sentence['terms']) > 0:
                all_data[doc.index].append(new_sentence)
    
    print("Number of distinct terms before cleaning:", raw_num_distince_terms)
    print("Number of distinct terms after cleaning:", clean_num_distince_terms)
    print("Number of sentences before cleaning:", raw_num_sentences)
    print("Number of sentences after cleaning:", clean_num_sentences)
    print("Number of unrelated and unannotated sentences:", unrelated_and_unannotated)
    print("Number of sentences that contain more than {} terms: {}".format(args.max_term_per_sent, too_many_terms))

    with open(args.save_file.replace(".json", str(args.max_term_per_sent) + ".json"), "w") as f:
        json.dump(all_data, f, indent= 4)

if __name__ == "__main__":
    args = get_args()
    main(args)
        