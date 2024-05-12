# -*- coding: utf-8 -*-

#importing library
import pandas as pd

#loading data
df = pd.read_csv('path')
df.head()


#checking shape
df.shape

#checking data types
df.dtypes


#checking for missing values
df.isnull().sum(axis = 0)


#concatenating columns
review_cols = ['Title',
               'Review Text',
               'Division Name',
               'Department Name',
               'Class Name']
#removing missing values using .fillna("")
df['Reviews'] = df[review_cols].fillna("").apply(lambda row: ' '.join(row.values.astype(str)), axis=1)


#checking data frame for new Reviews column
df.head()

#importing library
import re #regular expression for handling texts very efficiently

#cleaning the Reviews column using regular expressions
#removing punctuation and spaces; replacing special characters,<br /> in the file
df['Reviews'] = df['Reviews'].apply(lambda x: re.sub(r'[^A-Za-z0-9]+',' ',x))
df['Reviews'] = df['Reviews'].apply(lambda x: re.sub(r"<br />", " ", x))
# removing words with a length less than or equal to 2
df['Reviews'] = df['Reviews'].apply(lambda x: re.sub(r'\b[a-zA-Z]{1,2}\b', '', x))


df['Reviews'].head()

#importing library
from sklearn.model_selection import train_test_split

#performing train-test split
X = df['Reviews'].values
y = df['Recommended IND'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

#printing shape
print(f'X_train size={X_train.shape}; X_test size  ={X_test.shape}')

#importing library
import tensorflow as tf
import numpy as np

#specifying the vocab size
VOCAB_SIZE = 1000
#performing textvectorization
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)

#fitting the state of the preprocessing layer to the dataset.
encoder.adapt(X_train)

#building RNN model
model = tf.keras.Sequential([
    #performing textvectorization which converts the raw texts to indices/integers
    encoder,
    #embedding layer to convert the indices to numerical vectors
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        #using masking to handle the variable sequence lengths
        mask_zero=True),
    #GRU layer; the default recurrent_activation = sigmoid
    tf.keras.layers.GRU(256, return_sequences=True),
    #LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
    #classification layer 1
    tf.keras.layers.Dense(128, activation='relu'),
    #classification layer 2
    tf.keras.layers.Dense(64, activation='relu'),
    #classification layer 3; must be equal to 1 since this is the output layer
    tf.keras.layers.Dense(1, activation=None)
])

#summarizing model
model.summary()


#adding early stopping; if the validation accuracy does not improve for 3 epochs, we will stop training
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience= 3)


#configuring the model; since activation=None we must put from_logits=True
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy']) # we will evaluate the model using accuracy


# Commented out IPython magic to ensure Python compatibility.
# %%time
# #training the model and storing the history in a variable to plot later
# #we will make the number of epochs short since training models for image classification can run for a long time
# history = model.fit(x=X_train,
#                     y=y_train,
#                     epochs=10, #this can be time consuming so we will only run this for 10 epochs
#                     validation_data=(X_test,y_test),
#                     callbacks=[callback],
#                     verbose = 1)


#displaying model architecture
tf.keras.utils.plot_model(model, show_shapes=True)


#importing libraries
import seaborn as sns
import matplotlib.pyplot as plt

#visualizing training history
train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
#plotting train loss
sns.lineplot(x='epoch', y ='loss', data =train_history)
#plotting validation loss
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
#adding legends
plt.legend(labels=['train_loss', 'val_loss'])
plt.title('Training and Validation Loss Over Epochs')
plt.show()

#plotting training accuracy
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
#Plot validation accuracy
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
#adding legends
plt.legend(labels=['train_accuracy', 'val_accuracy'])
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()

#importing libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#the cutoff probability is 50%
y_pred = (model.predict(X_test)> 0.5).astype(int)

#creating confusion matrix
print(confusion_matrix(y_test, y_pred))

#forecasting the Recommended IND labels
#we need to convert 0 and 1 back to words so that 0 = Not Recommended and 1 = Recommended for easier readability
#printing a classification report
label_names = ['Not Recommended', 'Recommended']
print(classification_report(y_test, y_pred, target_names=label_names))

