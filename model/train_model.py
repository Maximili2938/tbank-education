# train_model.py
from models.qa_model import QAModel

def main():
    model = QAModel()
    train_loader = model.prepare_dataloader('sberquad.csv')  # !!  Убедитесь, что файл sberquad.csv находится в корне проекта !!
    model.train(train_loader)
    model.save()

if __name__ == '__main__':
    main()
