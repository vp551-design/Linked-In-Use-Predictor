LinkedIn Use Predictor

This project takes the Pew social media dataset and builds a logistic regression model to predict if someone uses LinkedIn based on basic demographic inputs. The Streamlit app lets you enter information and instantly see the prediction and probability, along with quick visuals based on the cleaned dataset.

What’s Included

app.py – the Streamlit app

social_media_usage.csv – dataset

requirements.txt – packages needed

README.md – project overview

What the App Does

Cleans the data

Trains the model

Lets you input income, education, parent status, marital status, gender, and age

Predicts LinkedIn use + shows the probability

Displays simple visuals for age, income, and education trends

Run the App
pip install -r requirements.txt
streamlit run app.py

Output

You’ll get:

Predicted LinkedIn user: Yes/No

Probability score

Updated visuals based on the dataset

Deployment

The app can be deployed on Streamlit Cloud using the same files in this repo.
