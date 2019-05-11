import functools
import json
import os
import re


MY_NAME = 'Minh Mai'


def validate_data(data):
    """Validate data before parsing message"""
    # ensures two participant in conversation,
    # no group chats or one way conversations
    if len(set([user['name'] for user in data['participants']])) != 2:
        return False
            
    # ensure some level of length and back and forth in conversations
    if len(data['messages']) < 4:
        return False

    return True


def normalize_content(message):
    """Normalize strings by removing special characters"""
    content = message.get('content', '')
    msg = re.sub(r'[^ a-zA-Z0-9]', '', content)
    return msg.lower()


def process_messages(data, other_person):
    """Process messages by user"""
    inputs, outputs, current_texts = [], [], []
    current_person = other_person

    for msg in data:

        if msg['sender_name'] == current_person:
            current_texts.append(normalize_content(msg))
        else:
            combined_text = '.'.join(current_texts)

            if current_person == MY_NAME:
                outputs.append(combined_text)
                current_person = other_person
            else:
                inputs.append(combined_text)
                current_person = MY_NAME

            current_texts = [normalize_content(msg)]
    
    return inputs, outputs


def create_inputs_outputs(root_dir):
    """Create the input and outputs for network"""
    all_inputs = []
    all_outputs = []

    for filez in os.listdir(root_dir):
        message_path = os.path.join(root_dir, filez, 'message.json')

        if not os.path.exists(message_path):
            continue

        with open(message_path, 'r') as fd:
            data = json.load(fd)
            
            if not validate_data(data):
                continue
        
        other_person = [user['name']
            for user in data['participants']
            if user['name'] != MY_NAME][0]
        messages = sorted(data['messages'], key=lambda k: k['timestamp_ms'])
        start_number = [msg['sender_name'] for msg in messages].index(other_person)
        inputs, outputs = process_messages(messages[start_number:], other_person)
        all_inputs.extend(inputs)
        all_outputs.extend(outputs)

    return all_inputs, all_outputs


def create_corpus(data):
    results = {}
    current_number = 0

    for datum in data:
        for word in datum.split():

            if word not in results:
                results[word] = current_number
                current_number += 1

    results['\t'] = current_number + 1
    results['\n'] = current_number + 2
    return results


def vectorize_words(corpus, list_of_sentences, vector_type='input'):
    results = []
    for sentence in list_of_sentences:
        indexed = [corpus[word] for word in sentence.split()]

        # adding start and stopping indicators for sequence
        if vector_type == 'output':
            indexed = [corpus['\t']] + indexed + [corpus['\n']]

        results.append(indexed)
    return results
