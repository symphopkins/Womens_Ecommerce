# Women's E-Commerce Reviews Sentiment Analysis

## Overview
This project focuses on sentiment analysis using a Recurrent Neural Network (RNN) model applied to a Women’s Clothing E-Commerce dataset. The objective is to forecast the Recommended IND based on customer reviews concatenated from various features. The model's performance is evaluated to determine its suitability for recommendation to the e-commerce company.

## Files Included
- `Womens_Ecommerce.py`: Python script containing the code implementation.
- `Womens_Ecommerce.ipynb`: Google Colab Notebook containing the detailed code implementation and explanations.
- `requirements.txt`: Text file listing the required Python packages and their versions.
- `LICENSE.txt`: Text file containing the license information for the project.

## Installation
To run this project, ensure you have Python installed on your system. You can install the required dependencies using the `requirements.txt` file.

## Usage
1. The dataset `Womens Clothing E-Commerce Reviews.csv` should be downloaded and loaded into memory.
2. Concatenate the columns `Title`, `Review Text`, `Division Name`, `Department Name`, and `Class Name` into a new feature column called `Reviews`.
3. Clean the `Reviews` column using regular expressions to remove special characters, punctuation, spaces, and words with lengths less than or equal to 2.
4. Split the data into training and test sets. Build an RNN model with specific architecture including Embedding, GRU, LSTM, and classification layers.
5. Train the model, evaluate its performance using validation data, and check its fit using visualization of training and validation loss and accuracy. Additionally, assess the model's performance on the test dataset using a confusion matrix and classification report.

## Data Source
The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews).

## License
 Creative Commons Attribution 4.0 International License
