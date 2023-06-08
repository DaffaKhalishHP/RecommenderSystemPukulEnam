# RecommenderSystemPukulEnam

## Project Description

Team formation for projects often requires significant time and effort from project managers. Various factors such as interpersonal relationships, individual capabilities, and current availability of team members need to be considered. PukulEnam, an AI consulting services company, also faces these challenges in their project management process. The team formation process involves checking individual availability, reviewing past project experiences, and ensuring a good fit for the client's needs.

PukulEnam offers a range of consulting services, including digital transformation, data transformation, and artificial intelligence technology development. As a result, the company has a diverse pool of talent with skills in website development (back-end and front-end), cloud engineering, technical writing, data science, and data analytics. Currently, there are 12 talented individuals in the company.

Due to the diversity of skills and varying availability of team members, team formation takes a significant amount of time and energy. To address this problem, our team has developed a Recommender System using machine learning. This system automates the team formation process and provides a compatibility percentage for each team member based on their current availability and the client's needed skill.


## Objective

The Team Formation Recommender System aims to automate the team formation process by considering factors such as past projects, individual skill sets, and availability. It offers the following functionalities:

1. Automated Team Formation: The system streamlines the team formation process by leveraging information from past projects, individual skill sets, and availability. This automation reduces the burden on project managers and saves time and effort.
2. Fit Estimation: The recommender system provides an estimate of how well each individual aligns with the client's requirements based on their skill sets. This assessment assists project managers in identifying the most suitable team members for a specific project.
3. Real-Time Availability Updates: The system enables project managers to easily update the availability status of each individual. As projects are taken or completed, the availability information can be promptly modified, facilitating efficient communication and coordination.

## Installation

## Data Collecting and Preprocessing
PukulEnam provides real past projects data, however after consideration our team decided to make a dummy data consist of 50.000 rows of past project data to train the model. The dataset comprises four columns of features: Topics, Subtopics, Difficulty, and Project Type. Additionally, there is a label column called Workers. But in the future development project manager can do transfer learning to fit real-life data for more accurate prediction.

Data Preprocessing can be achieve using some Python Library such as:
- Pandas (For pre-processing data from imported CSV)
- SKLearn Model_Selection (For splitting train,validation and test data)
- SKLearn Preprocessing (For Encoding the label into One-Hot Code)
- Numpy (For transform shape of data)
- Tensorflow (For Creating Model and Dense Layer)

### Steps of Preprocessing Data

1. First load the Data CSV and Remove the unused column 
```python
# Load The CSV
df = pd.read_csv(r".\datadummy50k_new_grouped.csv")
# Cleaning the unused columns
df = df.drop(df.columns[[0]], axis=1) 
```

2. Then We Transform the Workers from strings to list, and renaming the other columns
```python 
# Transform 'Workers' from strings into lists
df['Workers'] = df['Workers'].str.replace("[\'\[\]]","",regex=True)
df['Workers'] = df['Workers'].str.replace(", ","|",regex=True)
df['Workers'] = df['Workers'].apply(lambda s: [l for l in str(s).split('|')])

df.rename(columns={'Project Type': 'Project_Type'}, inplace=True)
df.rename(columns={'Sub Topic': 'Sub_Topic'}, inplace=True)

```

3. For encoding the strings column into interger we use enumerate whole categorical columns and transform it to interger, the other columns will be transform into strings

```python
top_dict = dict(enumerate(df["Topics"].astype('category').cat.categories))
subtop_dict = dict(enumerate(df["Sub_Topic"].astype('category').cat.categories))
ptype_dict = dict(enumerate(df["Project_Type"].astype('category').cat.categories))

# Transform columns from string into integer
string_col = ['Topics', 'Sub_Topic', 'Project_Type']
for col in string_col:
  df[col] = df[col].astype('category').cat.codes

# Transform other columns into strings
for col in string_col:
  df[col] = df[col].astype('category')


```
4. For Label column encoding, we use MultilabelBinazer to turn our label to One-hot Encoding 
```python
# Creating list of labels
labels_list = df['Workers']
labels_list = list(labels_list)
mlb = MultiLabelBinarizer()
mlb.fit(labels_list)

N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))
```

5. Splitting The data to train and validate
```python

# Making the labels

train, test = train_test_split(df, test_size=0.05)
train, val = train_test_split(train, test_size=0.05)

train_labels = train.pop('Workers')
val_labels = val.pop('Workers')
test_labels = test.pop('Workers')

train_labels = list(train_labels)
val_labels = list(val_labels)
test_labels = list(test_labels)

#Encode each Split Labels
train_labels2 = mlb.transform(train_labels)
val_labels2 = mlb.transform(val_labels)
test_labels2 = mlb.transform(test_labels)
```


