# Restaurant Review Sentiment Analysis

This project demonstrates sentiment analysis on restaurant reviews using Natural Language Processing (NLP) techniques in Python within a Google Colab environment.

## Project Overview

The goal of this project is to build and evaluate models that can accurately classify restaurant reviews as positive or negative based on their textual content. The project leverages a dataset of restaurant reviews and employs various NLP and machine learning techniques to achieve this goal.

## Dataset

The project utilizes the "Restaurant_Reviews.tsv" dataset, which contains a collection of restaurant reviews labeled as positive or negative. This dataset is commonly used for sentiment analysis tasks and provides a good benchmark for evaluating model performance.

## Methodology

The project follows a systematic approach to sentiment analysis, involving the following key steps:

1. **Data Cleaning and Preprocessing:**
   - The raw text data is cleaned by removing punctuation, converting to lowercase, and removing irrelevant words (stop words).
   - Stemming is applied to reduce words to their root form using the Porter Stemmer.

2. **Feature Engineering:**
   - The Bag-of-Words model is used to represent the text data numerically.
   - The `CountVectorizer` from scikit-learn is used to create a vocabulary of words and generate a matrix representing the frequency of each word in each review.

3. **Model Training and Evaluation:**
   - Various machine learning classifiers are trained and evaluated, including Naive Bayes, SVM, Random Forest, and Logistic Regression.
   - Model performance is assessed using the confusion matrix and accuracy score metrics.

4. **Prediction:**
   - The trained models are used to predict the sentiment of new, unseen restaurant reviews.
   - The prediction process involves transforming the new review text using the same preprocessing and feature engineering steps applied during training.
   - The model then assigns a sentiment label (positive or negative) based on the learned patterns from the training data.

## Results

The project achieves promising results in sentiment classification, with the best-performing models demonstrating high accuracy on the test set. The confusion matrix provides insights into the model's performance on different classes, while the accuracy score quantifies the overall correctness of the predictions.

## Usage

To run the project, follow these steps:

1. Clone the repository to your local machine.
2. Open the Jupyter Notebook file in Google Colab.
3. Execute the code cells in sequential order.
4. Modify the input text to predict the sentiment of new reviews.

## Contributing

Contributions to this project are welcome. Potential areas for improvement include:

- Exploring and implementing more advanced NLP techniques, such as word embeddings or deep learning models.
- Fine-tuning model parameters to optimize performance.
- Experimenting with different feature engineering methods.
- Adding visualizations to better understand the data and model behavior.
- Expanding the project to handle different datasets or languages.

## License

This project is licensed under the MIT License.

## Acknowledgments

This template for natural language processing is teached to me in the udemy course called "MACHINE LEARNING A TO Z course by Hadelin de Ponteves and Kirill Eremenko
