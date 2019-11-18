import spacy
import os
import numpy as np
import scipy.spatial.distance as distance
import gensim
from collections import defaultdict
import re
from src.preprocess import remove_punctuation

word2vec = gensim.models.KeyedVectors.load_word2vec_format(
    "../data/GoogleNews-vectors-negative300.bin", binary=True)
nlp = spacy.load('en_core_web_sm')

def word_weights(filepath):

    with open(filepath) as f:
        content = f.read()



def get_sentence_vector(sentence):
    print("Geting maximum sentence similarity of each sentence...")

    # Word embeddings GoogleNews-vectors-negative300.bin.gz from Word2Vec :
    # https://code.google.com/archive/p/word2vec/ as input for each word.


    # Calculate the average word embedding for sentences

    # use the maximum similarity as the features.

    word2vec_dict = word2vec.vocab.keys()

    wordvecs = []
    # sentence = nlp(sentence)
    for word in sentence:
        # Words in a summary that are not covered by Word2Vec are discarded.
        word = str(word).lower()
        if word in word2vec_dict:
            wordvecs.append(word2vec[word])

    wordvecs = np.array(wordvecs)
    # represent each sentence as average of its word embedding
    sentence_score = np.mean(wordvecs, axis=0)


    return sentence_score

#
def calculate_sentence_sim(s1):
    #       - Statement 1 (Clear Purpose): For what purposes does the company use personal information?
    #     # - Statement 2 (Third Parties): Does the company share my information with third parties?
    #     # - Statement 3 (Limited Collection): Does the company combine my information with data from other sources?
    #     # - Statement 4 (Limited Use): Will the company sell, re-package or commercialize my data?
    #     # - Statement 5 (Retention): Will the company retain my data? What is their retention policy?

    s  = get_sentence_vector(s1)
    c1 = get_sentence_vector(nlp(remove_punctuation("For what purposes does the company use personal information?")))
    c2 = get_sentence_vector(nlp(remove_punctuation("Does the company share my information with third parties?")))
    c3 = get_sentence_vector(nlp(remove_punctuation("Does the company combine my information with data from other sources?")))
    c4 = get_sentence_vector(nlp(remove_punctuation("Will the company sell, re-package or commercialize my data?")))
    c5 = get_sentence_vector(nlp(remove_punctuation("Will the company retain my data? What is their retention policy?")))

    res = [0] * 5
    res[0] = 1 - distance.cosine(s, c1)
    res[1] = 1 - distance.cosine(s, c2)
    res[2] = 1 - distance.cosine(s, c3)
    res[3] = 1 - distance.cosine(s, c4)
    res[4] = 1 - distance.cosine(s, c5)

    return res

def get_policy_vectors(filepath):

    with open(filepath,'r') as f:

        data = nlp(remove_punctuation(f.read()))
        res = {}
        for sentence in data.sents:
            sentence_sim_vector = calculate_sentence_sim(sentence)
            # if max(sentence_sim_vector) >= 0.65:
            #     print(sentence)
            res[sentence] = sentence_sim_vector

    return res
#
# def get_all_policy_vectors(src_path):
#     all_policies = {}
#     for file in os.listdir(src_path):
#         with open(src_path + '/'+ file, 'r') as f:
#             all_policies[file] = get_all_policy_vectors(src_path + '/'+ file)
#
#     return all_policies
#
#

similarity_array = get_policy_vectors('../data/notags_policies/20_theatlantic.txt')
class_array={}
for k,row in similarity_array.items():
    if max(row) >= 0.65:
        class_array[k] = row.index(max(row)) + 1
    else:
        class_array[k] = 0



# class_array = {k :(row.index(max(row)) + 1) if max(row) >= 0.65 else k:0 for k,row in similarity_array.items()}


# Group by values
res = defaultdict(list)
for key, val in sorted(class_array.items()):
    res[val].append(key)

class_names = {0:'Other',1:'Clear Purpose',2:'Third Parties',3:'Limited Collection',4:'Limited Use',5:'Retention'}

with open('../results/20_theatlantic.txt', 'w') as f:
    for i, val in res.items():
        f.write('\n\n' + 'Class is : '+ str(i) + ' ' + class_names[i] + '\n\n')
        for item in val:
            f.write(str(item)+'\n')

print(similarity_array)
print(class_array)