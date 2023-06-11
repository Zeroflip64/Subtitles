## [Определение сложности субтитров / Recognition of language complexity for understanding movie subtitles](https://github.com/Zeroflip64/Subtitles/blob/main/%D0%9C%D0%B0%D1%81%D1%82%D0%B5%D1%80%D1%81%D0%BA%D0%B0%D1%8F_2.ipynb)

[Ссылка на приложение/Link to the application](https://zeroflip64-subtitles-streamlit-sub-n9wsxl.streamlit.app/)

### Цель\Goal
Необходимо создать приложение где пользователь может загрузить субтитры к фильму  и получит уровень знания языка который необходим для понимания этого фильма  / It is necessary to create an application where the user can download subtitles for the movie and get the level of knowledge of the language that is necessary to understand this movie


### Описание проекта
Школа англйского языка заказала проект для того что бы легче определять уровень языка.Необходимо было разработать модель которая бы предсказывала уровень сложности вместе с этим предоставив небольшую информацию об этих субтитрахи так же было решено добавить возможность польщователю после определения уровня субтитров выбрать подходящие фильмы похожего уровня

ENG:
The English language School has commissioned a project to make it easier to determine the level of the language.It was necessary to develop a model that would predict the level of complexity along with providing a little information about these subtitles, it was also decided to add the ability to the user after determining the level of subtitles to select suitable films of a similar level.

### Project Description
Проект состоит из несколкьих этапов:
* Иначально был предоставлен набор данных с названием фильмов и их уровнем.
* Сбор дополнительных субтитров.
* Разработка класса с функциями которые могли бы преоставить дополнительную аналитику по субтитрам.
* Очистка и подготовка данныхб токенизация
* Разработка модели
* Внедрение в Streamlit

ENG: 

The project consists of several stages:
* Initially, a data set with the name of the films and their level was provided.
* Collecting additional subtitles.
* Development of a class with functions that could provide additional analytics on subtitles.
* Data cleaning and preparation, tokenization
* Model development
* Implementation in Streamlit

### Используемые библиотеки \ Libraries used
- **`nltk`**
- **`textstat`**
- **`pysrt`**
- **`Matplotlib`**
- **`spacy`**
- **`Catboost`**
- **`re`**
- **`tensorflow`**
### Результаты \ Results
Было разработано приложение которое удовлетваряло заказчика и все необходимые функции выполнены. 
