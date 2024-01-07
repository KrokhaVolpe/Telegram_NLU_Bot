import random
import nltk

import csv

fname = 'bot_config.csv'


with open(fname) as f:
    reader = csv.DictReader(f)
    print(reader.fieldnames)
    lines = list(reader)

codes = [l['Ячейка з BOT_CONFIG'] for l in lines]
print(len(codes))

nltk.edit_distance('Доброго дня', 'Доброго вечора')


def filter_text(text):
    text = text.lower()
    text = [c for c in text if c in 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя- ']
    text = ''.join(text)
    return text


def get_intent(question):
    for intent, intent_data in BOT_CONFIG['intents'].items():
        for example in intent_data['examples']:
            filtered_example = filter_text(example)
            dist = nltk.edit_distance(filtered_example, filter_text(question))
            if dist / len('filtered_example') < 0.4:
                return intent
        


def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        pharase = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(pharase)
    
    
def generate_answer_by_text(question):
    return #TODO


def get_failure_pharase():
    pharase = BOT_CONFIG['failure_pharase']
    return random.choice(pharase)

def bot(question):
    #NLU
    intent = get_intent(question)
    
    #getting a response
    
    #looking for a ready answer
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            return answer
    
    #generate a response that is appropriate for the context
    answer = generate_answer_by_text(question)
    if answer:
            return answer
    
    #let's use a stub
    answer = get_failure_pharase()
    return  answer


question = None

while question not in ['exit', 'вихід']:
    question = input()
    answer = bot(question)
    print(answer)
