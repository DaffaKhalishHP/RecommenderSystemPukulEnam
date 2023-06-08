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






