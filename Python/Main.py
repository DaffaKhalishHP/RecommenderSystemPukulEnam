import pandas as pd
import tensorflow as tf
import gdown
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Preparing the pandas dataframe

id = "1a6TppDeLhFhKso9vkYzRdPfualSRmRFf"
output = "datadummy50k_new_grouped"
gdown.download(id=id, output=output, quiet=False)

file_id = "1a6TppDeLhFhKso9vkYzRdPfualSRmRFf"
output_file = r".\datadummy50k_new_grouped.csv"  # Specify the full path

gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file, quiet=False)

# Your code to work with the downloaded file goes here


# ganti directorynya 
df = pd.read_csv(r".\datadummy50k_new_grouped.csv")

# Cleaning the unused columns
df = df.drop(df.columns[[0]], axis=1)

# Transform 'Workers' from strings into lists
df['Workers'] = df['Workers'].str.replace("[\'\[\]]","",regex=True)
df['Workers'] = df['Workers'].str.replace(", ","|",regex=True)
df['Workers'] = df['Workers'].apply(lambda s: [l for l in str(s).split('|')])

# for checking the data
df

# Create dictionaries
# Assuming you have a DataFrame called 'df' and you want to rename the column 'old_column' to 'new_column'
df.rename(columns={'Project Type': 'Project_Type'}, inplace=True)
df.rename(columns={'Sub Topic': 'Sub_Topic'}, inplace=True)


top_dict = dict(enumerate(df["Topics"].astype('category').cat.categories))
subtop_dict = dict(enumerate(df["Sub_Topic"].astype('category').cat.categories))
ptype_dict = dict(enumerate(df["Project_Type"].astype('category').cat.categories))

#checking
print(top_dict, subtop_dict, ptype_dict)


string_col = ['Topics', 'Sub_Topic', 'Project_Type']

# Transform columns from string into integer

for col in string_col:
  df[col] = df[col].astype('category').cat.codes

df.head()

# Transform other columns into strings

for col in string_col:
  df[col] = df[col].astype('category')

print(df.dtypes)
df.head()

# Creating list of labels
labels_list = df['Workers']
labels_list = list(labels_list)
mlb = MultiLabelBinarizer()
mlb.fit(labels_list)

N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))

#splitting the data 

train, test = train_test_split(df, test_size=0.05)
train, val = train_test_split(train, test_size=0.05)

# checking the size

train_size = len(train)
val_size = len(val)
test_size = len(test)

print("Number of examples in the train set:", train_size)
print("Number of examples in the validation set:", val_size)
print("Number of examples in the test set:", test_size)

# Making the labels

train_labels = train.pop('Workers')
val_labels = val.pop('Workers')
test_labels = test.pop('Workers')

train_labels = list(train_labels)
val_labels = list(val_labels)
test_labels = list(test_labels)

train_labels2 = mlb.transform(train_labels)
val_labels2 = mlb.transform(val_labels)
test_labels2 = mlb.transform(test_labels)

# Custom F1Score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        f1_score = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        return f1_score

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# get the model
def get_model(n_inputs, n_outputs):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(1000, input_dim=n_inputs,use_bias=True, kernel_initializer='he_uniform', activation='relu'))
	model.add(tf.keras.layers.Dense(23, use_bias=True, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=tf.keras.metrics.Precision())
	return model

#Getting the shape of data 
n_inputs, n_outputs = train.shape[1], train_labels2.shape[1]
# get model
model = get_model(n_inputs, n_outputs)

#Train the data

# using val set
model.fit(x=train, y=train_labels2, validation_data=(val, val_labels2), epochs=20, verbose=1)

# Evaluating the model using the test set
loss, accuracy = model.evaluate(x=test, y=test_labels2)
print("Accuracy", accuracy)


# Saving The Model

import tensorflow as tf

model.save(r'.\model', save_format='tf')

# Load the model
model = tf.keras.models.load_model(r'.\model')

# compile the model to another format
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=tf.keras.metrics.Precision()
)

model.save(r'.\model.h5')