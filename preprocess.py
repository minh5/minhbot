import os
import json


def create_words_list(root_dir):
    results = []
    for filez in os.listdir(root_dir):
        message_path = os.path.join(root_dir, filez, 'message.json')
        if not os.path.exists(message_path):
            continue
        with open(message_path, 'r') as fd:
            data = json.load(fd)
        for msg in data['messages']:
            if msg.pop('sender_name', None) == 'Minh Mai':
                content = msg.pop('content', '')
                content = content.split()
                if len(content) > 5:
                    results.append(content)
    return results


def create_corpus(list_of_sentences):
    results = {}
    counter = 0
    for sentence in list_of_sentences:
        for word in sentence:
            idx = word.lower().strip()
            if idx not in results:
                results[idx] = counter
                counter += 1
    return results


def words_to_idx(list_of_sentences, corpus):
    max_length = 
    results = []
    for sentence in list_of_sentences:
        [corpus[word.lower().strip()] for word in sentence]
        results.append()
    return results