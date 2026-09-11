Сервис классификации товаров H&M

Проект предназначен для классификации текстовых описаний товаров по категориям с использованием методов машинного обучения.
В проекте сравниваются две модели: Logistic Regression + TF-IDF и LSTM 
Для предоставления предсказаний разработан REST API на FastAPI.

В сервисе FastAPI используется модель Logistic Regression + TF-IDF, т.к. данная модель показала более точные результаты предсказаний. Сравнение метрик см. ниже. 

Классификация товаров выполняется по следующим категориям:
-- Jersey Fancy
-- Accessories
-- Jersey Basic
-- Knitwear
-- Under-, Nightwear

Структура проекта:
Final project/ 

├── app/ 

│ ├── __init__.py 

│ ├── main.py 

│ ├── model.py

│ └── schemas.py 

│ 

├── models/

│ ├── logistic_pipeline.pkl 

│ ├── lstm.pt 

│ └── tokenizer.pkl 

│ 

├── reports/ 

│ ├── class_distribution.png

│ ├── confusion_matrix_logistic.png 

│ ├── confusion_matrix_lstm.png 

│ ├── lstm_loss.png 

│ ├── mean_text_length_byclasses.png 

│ └── text_length_distribution.png 

│ 

├── src/ 

│ ├── __init__.py 

│ ├── data.py 

│ ├── eda.py 

│ ├── train_dummy.py 

│ ├── train_logistic.py 

│ ├── train_lstm.py 

│ └── evaluate.py 

│ 

├── tests/ 

│ └── test_api.py 

│ 

├── requirements.txt 

└── README.md


Требования: python 3.13 

--Установка--

1. Клонировать репозиторий или скачать проект и перейти в его корневую директорию cd "Final project"
2. Создать виртуальное окружение 
3. Активировать виртуальное окружение
4. Установить зависимости из файла requirements

-- Запуск и использование Fast API--

Команды выполняются из корневой директории

Запустить API: python -m uvicorn app.main:app --reload --port 8001
После запуска API будет доступно по адресу http://127.0.0.1:8001

Интерактивная документация Swagger UI доступна по адресу http://127.0.0.1:8001/docs
Альт. документация ReDoc: http://127.0.0.1:8001/redoc

--Проверка состояния API доступна через endpoint GET /health

Пример ответа:
{
    "status": "ok"
}

--Для получения предсказания используется endpoint POST /predict.

Пример входных и выходных данных:
{ 
    "prod_name": "Slim fit trousers", 
    "detail_desc": "Black trousers in woven fabric" 
}

{ 
    "predicted_group": "Jersey Basic", 
    "probabilities": { 
        "Accessories": 0.01, 
        "Jersey Basic": 0.85, 
        "Jersey Fancy": 0.08, 
        "Knitwear": 0.03, 
        "Under-, Nightwear": 0.03 
    } 
}

-- Для запуска автоматических тестов используется pytest, из корневой директории проекта выполнить python -m pytest 
Тесты проверяют:
1. Функцию предсказания
2. Корректность вероятностей
3. endpoint /health
4. enpoint /predict
5. Предсказания для различных товаров
6. Обработку некорректного ввода
7. Обработку пустого текста
Ожидаемый результат: 9 passed

--Обучение моделей--
Для обучения базовой модели DummyClassifier: python -m src.train_dummy
Для обучения Logistic Regression: python -m src.train_logistic
Для обучения LSTM: python -m src.train_lstm

Для оценки и сравнения моделей: python -m src.evaluate
Результаты обучения и оценки сохраняются в директории reports/

--Результаты моделей--

Для сравнения моделей используюся метрики Accuracy и Macro-F1

Logistic Regression
Accuracy: 0.9265
Macro-F1: 0.9240

LSTM
Accuracy: 0.8580
Macro-F1: 0.8485
