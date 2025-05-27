# 🏔️ Mountains vs Beaches Preference Predictor 🏖️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📌 Overview

This project is a machine learning application that predicts whether a person prefers **Mountains** or **Beaches** for their vacation based on demographic information and personal preferences. Created as a final project for Data Science Batch 47 at Digital Skola.

## 🎯 Features

- Interactive web application built with Streamlit
- Predicts vacation preferences with high accuracy (87.57%)
- Uses XGBoost classification model
- Real-time predictions based on user inputs
- Confidence score for each prediction
- Visualized probability distribution

## 🚀 Demo

Try the live demo: [Mountains vs Beaches Predictor](https://your-app-url.streamlit.app)

## 📊 Dataset

The model was trained on a dataset containing 52,444 records with the following features:

### Demographic Features:
- **Age**: Age of the individual
- **Gender**: Gender identity (male, female, non-binary)
- **Income**: Annual income in USD
- **Education Level**: Highest education attained
- **Location**: Type of residence (urban, suburban, rural)

### Travel Preferences:
- **Travel Frequency**: Number of vacations per year
- **Vacation Budget**: Budget allocated for vacations
- **Preferred Activities**: Activities during vacations
- **Favorite Season**: Preferred season for vacations

### Other Factors:
- **Proximity to Mountains**: Distance from nearest mountains (miles)
- **Proximity to Beaches**: Distance from nearest beaches (miles)
- **Pets**: Pet ownership (Yes/No)
- **Environmental Concerns**: Environmental awareness (Yes/No)

## 🤖 Model Performance

- **Algorithm**: XGBoost Classifier
- **Accuracy**: 87.57%
- **ROC AUC**: 92%
- **Precision**: 0.67
- **Recall**: 0.99
- **F1 Score**: 0.80

## 🛠️ Technologies Used

- **Python 3.9+**
- **Streamlit** - Web application framework
- **XGBoost** - Machine learning model
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Data preprocessing and metrics
- **Matplotlib/Seaborn** - Data visualization

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mountains-vs-beaches.git
cd mountains-vs-beaches
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
mountains-vs-beaches/
│
├── app.py                                    # Streamlit application
├── Vacation_Preference_XGBoost_Model.pkl     # Trained model
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
│
└── notebooks/
    └── Final_Project_Tes3.ipynb            # Jupyter notebook with EDA and model training
```

## 📈 Model Development Process

1. **Exploratory Data Analysis (EDA)**
   - Data distribution analysis
   - Feature correlation analysis
   - Target variable balance check

2. **Data Preprocessing**
   - Label encoding for ordinal features
   - One-hot encoding for categorical features
   - Feature scaling with StandardScaler

3. **Model Selection**
   - Compared 6 different algorithms
   - Selected XGBoost based on performance metrics
   - Cross-validation for robust evaluation

4. **Model Deployment**
   - Saved model using pickle
   - Created interactive Streamlit interface
   - Deployed on Streamlit Cloud

## 👥 Team Members - Group 10

- Aswar Hanif
- Ivander Perdana Mokhtar
- Rifqi Okta Saputra
- Wahyu Wahid

## 📝 License

This project is created for educational purposes as part of Digital Skola Data Science Batch 47.

## 🙏 Acknowledgments

- Digital Skola for providing the learning platform
- Instructors and mentors for guidance
- Dataset source: [Kaggle - Mountains vs Beaches Preference](https://www.kaggle.com/datasets/jahnavipalival/mountains-vs-beaches-preference)

## 📧 Contact

For questions or feedback about this project, please contact the team members.

---

**Note**: This is a student project created for learning purposes. The predictions are based on patterns in the training data and should not be used for critical decision-making.