# bot.py
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import requests

TOKEN = 'ВАШ_TELEGRAM_BOT_TOKEN'  # Замените на токен вашего бота

def start(update: Update, context: CallbackContext):
    update.message.reply_text('Привет! Задай мне вопрос.')

def handle_message(update: Update, context: CallbackContext):
    text = update.message.text
    # Предполагаем, что контекст совпадает с вопросом
    response = requests.post('http://localhost:5000/qa', json={'question': text, 'context': text})
    if response.status_code == 200:
        answer = response.json().get('answer', 'Ответ не найден.')
    else:
        answer = 'Произошла ошибка при обработке запроса.'
    update.message.reply_text(answer)

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
