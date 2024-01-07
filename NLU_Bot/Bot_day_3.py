import nltk

with open('dialogues.txt', encoding='utf-8') as f:
    content = f.read()

dialogues = [dialogue_line.split('\n') for dialogue_line in content.split('\n\n')]


def filter_text(text):
    text = text.lower()
    text = [c for c in text if c in 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя- ']
    text = ''.join(text)
    return text

questions = set()
qa_dataset = []  #[[q, a], ...]

for replicas in dialogues:
    if len(replicas) < 2:
        continue
    
    question, answer = replicas[:2]
    question = filter_text(question[2:])
    answer = answer[2:]

    if  question and  question not in questions:
        questions.add(question)
        qa_dataset.append([ question, answer])


qa_by_word_dataset = {}  # {'word': [[q, a], ...]}
for  question, answer in qa_dataset:
    words = question.split(' ')
    for word in words:
        if word not in qa_by_word_dataset:
            qa_by_word_dataset[word] = []
        qa_by_word_dataset[word].append((question, answer))

qa_by_word_dataset_filtered = {word: qa_list
                               for word, qa_list in qa_by_word_dataset.items()
                               if len(qa_list) < 1500}       

def generate_answer_by_text(text):
    text = filter_text(text)
    words = text.split(' ')
    qa = []
    for word in words:
        if word in qa_by_word_dataset_filtered:
            print(len(qa_by_word_dataset_filtered[word]))
            qa += qa_by_word_dataset_filtered[word]
    
    qa = set(qa)

    results = []
    for question, answer in qa:
        dist = nltk.edit_distance(question, text)
        dist_percentage = dist / len(question)
        results.append([dist_percentage, question, answer])
        
    if results:
        dist_percentage, question, answer = min(results, key=lambda pair: pair[0])

        if dist_percentage < 0.2:
            return answer
    return None
    
print(generate_answer_by_text('я тебя люблю?'))

