from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, Filters, ContextTypes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
import random
import nltk
import os
import asyncio

BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Привет', 'Добрый день', 'Приветствую', 'Здравствуйте', 'Приветствие', 'Доброе утро', 'Добрый вечер!', 'Добрый день!', 'Хеллоу!', 'Привет!', 'HI'],
            'responses': ['Приветствую! Как я могу вам помочь?', 'Здравствуйте! Очень приятно вас видеть!']
        },
        'bye': {
            'examples': ['До свидания', 'До встречи', 'Прощай', 'Спокойной ночи', 'Goodbye', 'Пока', 'До скорой встречи', 'Спасибо'],
            'responses': ['До свидания!', 'До встречи! Пусть вам повезет!', 'Goodbye! Have a great day!', 'До скорой встречи! Буду здесь, если вам нужна помощь.']
        },
        'help': {
            'examples': ['Помогите', 'Нужна помощь', 'Я не понимаю', 'Что делать в этом боте?'],
            'responses': ['Конечно, я готов вам помочь. Если у вас есть вопросы, просто спросите.', 'Я готов предоставить необходимую информацию. Как я могу помочь?']
        },
        'greeting': {
            'examples': ['Как дела?', 'Как вы чувствуете себя?', 'Приветствую вас', 'Как вы?', 'Всем привет!', 'Приветствие!'],
            'responses': ['Спасибо за вопрос, все хорошо. Как я могу вам помочь?', 'Очень хорошо, спасибо. Как я могу вам помочь?']
        },
        'wikipedia': {
            'examples': ['Что такое машинное обучение?', 'Кто такой Илон Маск?', 'Как работает солнечная энергия?', 'История Римской империи'],
            'responses': ["Машинное обучение - это область искусственного интеллекта, которая изучает, как компьютеры могут учиться выполнять различные задачи без явного программирования.",
                          'Илон Маск - известный предприниматель и основатель компаний SpaceX и Tesla, работающий над развитием космических технологий и автономных автомобилей.',
                          'Солнечная энергия генерируется с использованием солнечных панелей, которые преобразуют солнечное излучение в электроэнергию.',
                          'Римская империя была древней империей, существовавшей примерно тысячу лет и охватывавшей многие части Европы, Азии и Африки.']
        },
        'news': {
            'examples': ['Последние новости', 'Свежие новости', 'Новости о технологиях', 'Политические новости'],
            'responses': ['К сожалению, я не могу предоставлять актуальные новости. Рекомендую проверить соответствующие новостные источники для последних обновлений.']
        },
        'history': {
            'examples': ['История Украины', 'Средневековье', 'Вторая мировая война', 'Большой взрыв'],
            'responses': ['История Украины во многом интересна и сложна. В зависимости от вашего запроса, я могу поделиться кратким обзором исторических событий.',
                          'Средневековье - это эпоха в истории, которая включает множество важных событий и культурных изменений.',
                          'Вторая мировая война была крупнейшей войной в истории, которая изменила мировой ландшафт и оказала серьезное влияние на события.',
                          'Большой взрыв был важным открытием в физике и оказал значительное влияние на развитие науки.']
        },
        'books': {
            'examples': ['Рекомендации по книгам', 'Что почитать?', 'Любимые авторы'],
            'responses': ['Книги - это источник незабываемых приключений. Я могу порекомендовать вам несколько интересных книг для чтения.']
        },
        'sports': {
            'examples': ['Спорт', 'Футбол', 'Баскетбол', 'Фитнес', 'Здоровый образ жизни'],
            'responses': ['Спорт - это важный аспект здорового образа жизни. Я могу рассказать о различных видах спорта и тренировках.']
        },
        'movies': {
            'examples': ['Рекомендации по фильмам', 'Последние киноновости', 'Фильмы с любимыми актерами'],
            'responses': ['Фильмы - это прекрасный способ развлечься. Я могу порекомендовать вам несколько интересных фильмов для просмотра.']
        },
    },
    'failure_phrases': [
        'Извините, я еще учусь и не понимаю этот запрос. Попробуйте другой запрос, пожалуйста.',
        'Моя база данных не содержит информации по этому запросу. Возможно, я смогу вам помочь с другой темой?',
        'К сожалению, на данный момент я не могу понять ваш запрос. Возможно, у меня есть другие полезные функции, которые я могу выполнить.',
        'Это задание вне моих возможностей, но я всегда буду стараться учиться и улучшаться.',
        'Похоже, я попал в лес. Возможно, вы можете задать мне другой запрос?'
    ]
}



X_texts = []
y = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_texts.append(example)
        y.append(intent)

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
X = vectorizer.fit_transform(X_texts)
clf = LinearSVC(dual=False).fit(X, y)

# ваша функция filter_text теперь единая
def filter_text(text):
    text = text.lower()
    text = [c for c in text if c in 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя- ']
    text = ''.join(text)
    return text

# ваша функция generate_answer_by_text теперь разделяется на две функции
def generate_answer_by_text_using_similarity(text):
    text_vectorized = vectorizer.transform([text])

    similarities = cosine_similarity(text_vectorized, X).flatten()
    best_match_index = similarities.argmax()

    if similarities[best_match_index] > 0.2:
        return BOT_CONFIG['intents'][y[best_match_index]]['responses'][0]

def generate_answer_by_text_from_qa_dataset(text):
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


def get_intent(question):
    question_vector = vectorizer.transform([question])
    intent = clf.predict(question_vector)[0]

    examples = BOT_CONFIG['intents'][intent]['examples']
    for example in examples:
        dist = nltk.edit_distance(question, example)
        dist_percentage = dist / len(example)
        if dist_percentage < 0.4:
            return intent

    index = list(clf.classes_).index(intent)
    if clf.decision_function(question_vector)[0][index] > 0.5:  
        return intent

# ваша функция get_failure_pharase остается неизменной
def get_failure_pharase():
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)

# ваша функция bot теперь использует две новые функции generate_answer_by_text
def bot(question):
    # NLU
    intent = get_intent(question)

    # getting a response

    # looking for a ready answer
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats[0] += 1
            return answer

    # generate a response that is appropriate for the context
    answer = generate_answer_by_text_using_similarity(question)
    if answer:
        stats[1] += 1
        return answer

    # let's use a stub
    stats[2] += 1
    answer = get_failure_pharase()
    return answer




#6899735748:AAFNliWbby32orf8jD2dIjLq6ZFjOJipYL0

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('HI!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Help!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    answer = bot(update.message.text)
    await update.message.reply_text(answer)
    print(stats)
    print("-", update.message.text)
    print("-", answer)
    print()

def main() -> None:
    """Start the bot."""
    application = Application.builder().token("6899735748:AAFNliWbby32orf8jD2dIjLq6ZFjOJipYL0").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(Filters.TEXT & ~Filters.COMMAND, echo))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    asyncio.run(main())
