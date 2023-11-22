# Mushroom Classification Web App

Binary Classification Web App to determine if mushrooms are edible or poisonous.

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

## Dependencies

- streamlit==0.88.0
- pandas==1.3.3
- numpy==1.21.2
- matplotlib==3.4.3
- scikit-learn==0.24.2
