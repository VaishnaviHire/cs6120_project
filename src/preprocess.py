import os
from bs4 import BeautifulSoup
import re

def convert_to_text(src_dir, target_dir):
    for file in os.listdir('../data/original_policies'):
        with open('../data/original_policies' + '/' + file, 'r', encoding="ISO-8859-1") as f:
            print(file)
            data = f.read()
            # print(data)
            bs = BeautifulSoup(data,'html.parser')
            texts = bs.findAll(['title', 'body','p','strong'])

        with open('../data/clean_policies' + '/' + file, 'w') as f:
            for t in texts:
                f.write(t.text)

def convert_clean_summaries(src_dir, target_dir):
    for file in os.listdir('../data/sanitized_policies'):
        with open('../data/sanitized_policies' + '/' + file, 'r', encoding="ISO-8859-1") as f:

            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, '', f.read())
            filename = file.split('.', -1)[0] + '.txt'

        with open('../data/notags_policies' + '/' + filename, 'w') as f:
            f.write(cleantext)


def remove_punctuation(data):
    data = re.sub("_", "", data)
    data = re.sub("[^\w\s]", "", data)
    data = re.sub(' +', ' ', data)

    return data


