# app.py
from flask import Flask, request, jsonify
from models.qa_model import QAModel
import torch

app = Flask(__name__)

# Инициализация модели
model = QAModel()
model.load()

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    question = data.get('question')
    context = data.get('context')

    if not question or not context:
        return jsonify({'error': 'Пожалуйста, предоставьте и вопрос, и контекст.'}), 400

    inputs = model.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

    answer = model.tokenizer.convert_tokens_to_string(model.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
