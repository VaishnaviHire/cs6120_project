import spacy
import os
import numpy as np
import scipy.spatial.distance as distance
import gensim
from collections import defaultdict
import re
from src.preprocess import remove_punctuation
from sklearn.neighbors import KNeighborsClassifier
import math


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

def get_policy_vectors_sents(filepath):

    with open(filepath,'r') as f:

        data = nlp(remove_punctuation(f.read()))
        res = {}
        for sentence in data.sents:

            sentence_sim_vector = get_sentence_vector(nlp((str(sentence.text))))

            if np.isnan(sentence_sim_vector).any():
                res[sentence] = np.array([0.0]*300)
                continue
            print(type(sentence_sim_vector))
            # if math.isnan(sentence_sim_vector):
            #     continue
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



similarity_array = get_policy_vectors('../data/notags_policies/33_nbcuniversal.txt')
vector_array = get_policy_vectors_sents('../data/notags_policies/33_nbcuniversal.txt')

class_array= {}
for k,row in similarity_array.items():
    if max(row) >= 0.65:
        class_array[k] = row.index(max(row)) + 1
    else:
        class_array[k] = 0




# x = list(vector_array.keys())
# y = list(class_array.keys())

# X = np.array([[float(x) for x in np.nditer(vector_array[x[i]])]for i in range(len(class_array.keys()))])
#
# for i in range(len(class_array.keys())):
#     Y.append(class_array[y[i]])

X = list(vector_array.values())
Y = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '2', '0', '45', '0', '34', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1234', '2', '4', '0', '2', '0', '0', '0', '1', '5', '5', '1', '1', '2', '0', '0', '1', '1', '2', '34', '0', '4', '4', '4', '4', '4', '12', '1', '2', '2', '2', '2', '23', '34', '3', '34', '2', '0', '13', '3', '34', '0', '24', '24', '24', '2', '3', '0', '2', '2', '2', '2', '2', '2', '2', '2', '0', '0', '0', '0', '0', '0', '0', '5', '2', '0', '0', '0', '5', '0', '0', '5', '5', '5', '5', '5', '5', '0', '1', '5', '15', '0', '0', '0', '0', '5', '5', '0', '0', '0', '0', '0', '15', '4', '5', '0', '0', '5', '5', '5', '0', '1', '24', '2', '2', '2', '2', '2', '0', '0', '0', '0', '1', '0', '0', '0', '0', '2', '2', '0', '2', '2', '2', '0', '0', '4', '4', '0', '0', '4', '2', '2', '0', '0', '0', '12', '2', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']

print(len(X), len(Y))
# for k, v in vector_array.items():
#     X.append(v)
    # Y.append(class_array[str(k)])

# class_array = {k :(row.index(max(row)) + 1) if max(row) >= 0.65 else k:0 for k,row in similarity_array.items()}


# # Group by values
# res = defaultdict(list)
# for key, val in sorted(class_array.items()):
#     res[val].append(key)
#
# class_names = {0:'Other',1:'Clear Purpose',2:'Third Parties',3:'Limited Collection',4:'Limited Use',5:'Retention'}
#
# with open('../results/knearest/20_theatlantic.txt', 'w') as f:
#     for i, val in res.items():
#         f.write('\n\n' + 'Class is : '+ str(i) + ' ' + class_names[i] + '\n\n')
#         for item in val:
#             f.write(str(item)+'\n')
#
# print(similarity_array)
# print(class_array)


#
# >>> X = [[0], [1], [2], [3]]
# >>> y = [0, 0, 1, 1]


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y)
# KNeighborsClassifier(...)
# print(neigh.predict([get_sentence_vector('When you choose to register for an account on any NBCUniversal online service the information you provide may be shared with NBCUniversal affiliates and used to help us better tailor providers carriers andor other mobile apps either operated by us or third parties')]))
# # print(neigh.predict([get_sentence_vector(nlp(remove_punctuation('Our goal is to collect and use information to deliver effective and personalized services which take your interests and preferences into account, as well as for our legitimate business needs and interests.')))]))
# # # # [0]
# # >>> print(neigh.predict_proba([[0.9]]))
# [[0.66666667 0.33333333]]

# 5,0
# print(neigh.predict([get_sentence_vector(nlp(remove_punctuation('This Privacy Policy explains how we collect use disclose and transfer the information you provide when you interact with us including but not limited to via our websites')))]))
# # 1,1
# print(neigh.predict([get_sentence_vector(nlp(remove_punctuation('Our goal is to collect and use information to deliver effective and personalized services which take your interests and preferences into account as well as for our legitimate business needs and interests.')))]))
#
# # Class: Third Parties
# # 3, 3
# print(neigh.predict([get_sentence_vector(nlp(remove_punctuation('Please see our Mobile Apps section for additional detail Information we share with advertisers including Targeted Advertising')))]))
# #
# print(neigh.predict([get_sentence_vector(nlp(remove_punctuation('These third parties may view edit or set their own tracking technologies cookies')))]))
#
# # Class: Limited Collection of Information
# print(neigh.predict([get_sentence_vector(nlp(remove_punctuation('This policy also tells you how we use tracking technologies cookies and browsing data we collect from your use of the online services the measures we take to protect the security of the information you provide to us through the online services and how you can contact us if you have any questions regarding the online services including if you want to unsubscribe from our services or update your contact details')))]))
# print(neigh.predict([get_sentence_vector(nlp(remove_punctuation('There are two main types of information we collect about users of our online services that include but are not limited to the following Information that identifies you')))]))
#
# # Class: Limited Usage of Information
#
#
# # Class: Information Retention
# print(neigh.predict([get_sentence_vector(nlp(remove_punctuation('This policy only applies to those online services that link to this policy')))]))
# print(neigh.predict([get_sentence_vector(nlp(remove_punctuation('By using the online services you expressly consent to our collection use disclosure and retention of your personal information as described in this Privacy Policy')))]))
