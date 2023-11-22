# Mushroom Classification Web App

Binary Classification Web App to determine if mushrooms are edible or poisonous.

[Link to Mushroom Classification Web App](https://mushroom-classification-app.streamlit.app/), deployed on Streamlit

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Description

This web app is designed to classify mushrooms as either edible or poisonous based on various features. Users can choose between different classifiers (Logistic Regression, Support Vector Machine, Random Forest) and visualize the classification results.

## Features

- Multiple classifiers to choose from.
- Visualization of classification metrics (Confusion Matrix, ROC Curve, Precision-Recall Curve).
- User-friendly interface to interact with the model.
- Raw data display option to explore the dataset.

## Getting Started

To run the web app locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YONGCACO3/Mushroom-Classification-App.git
   cd mushroom-classification-web-app
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

4. **Open the provided link in your web browser.**

## Usage

1. Choose a classifier from the sidebar.
2. Adjust hyperparameters if necessary.
3. Click the "Classify" button to see the results.
4. Explore additional metrics by selecting checkboxes.

## Screenshots

![Web App Screenshot](/web_app_screenshot.png)
![Web App Screenshot 2 ](/web_app_screenshot2.png)

## Dependencies

- streamlit==1.28.2
- pandas==2.0.3
- numpy==1.24.3
- matplotlib==3.7.2
- scikit-learn==1.3.2
