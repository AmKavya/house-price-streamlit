# house-price-streamlit

# 🏠 House Price Prediction – End-to-End ML App

An end-to-end **machine learning project** that predicts house prices using housing and neighborhood features.  
The project covers the complete ML lifecycle — data analysis, feature engineering, model training, evaluation, and deployment as an interactive **Streamlit web application**.

🔗 **Live Demo:**  
https://house-price-app-cmttybkjpeafe4b8e2qxog.streamlit.app/

---

## 📌 Project Overview

This project demonstrates how a machine learning model can be taken from a notebook environment to a real-world, user-facing web application.

The goal is to predict the **median value of owner-occupied homes (MEDV)** based on important housing characteristics while keeping the user interface clean and interpretable.

---

## 🧠 Dataset

- **Dataset:** Boston Housing Dataset  
- **Target Variable:** `MEDV` – Median house price (in $1000s)  
- **Features:** Crime rate, number of rooms, zoning information, socio-economic indicators, etc.

---

## 🔬 Notebook Workflow (`Linear.ipynb`)

The notebook implements the full machine learning pipeline:

### Data Exploration
- Studied feature definitions and distributions
- Used histograms and scatter plots for visualization
- Performed correlation analysis to identify influential features

### Train–Test Splitting
- Implemented random train–test splitting
- Applied **Stratified Shuffle Split** to preserve important categorical distributions (e.g., `CHAS`)
- Learned how stratification helps avoid sampling bias

### Feature Engineering
- Created engineered features such as **TAXRM (tax per room)**
- Selected meaningful features based on correlation and domain understanding
- Cleanly separated features and target variable

### Preprocessing Pipeline
- Built a reusable preprocessing pipeline using:
  - Median imputation for missing values
  - Feature scaling using `StandardScaler`
- Ensured consistency between training and inference

### Model Training & Evaluation
- Trained regression models (Linear Regression / Random Forest)
- Evaluated performance using **cross-validation**
- Used **RMSE** as the primary evaluation metric

### Model Persistence
- Saved the trained model and preprocessing pipeline using `joblib`
- Enabled seamless reuse during deployment

---

## 🖥️ Streamlit Web Application

The trained model is deployed as an interactive web app using **Streamlit**.

### User-Controlled Features

| Feature | Description |
|-------|------------|
| RM | Average number of rooms per dwelling |
| CRIM | Per capita crime rate by town |
| LSTAT | Percentage of lower status population |
| ZN | Proportion of residential land zoned for large lots |
| CHAS | Charles River proximity (Yes / No) |

Other features are held at statistically reasonable default values based on the training data.

---

## ✨ Application Features

- Clean and responsive UI
- Sliders and dropdowns for user input
- Real-time house price prediction
- Advanced assumptions hidden for simplicity
- Publicly accessible deployment (no local setup required)

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Joblib  
- Matplotlib, Seaborn  
- Streamlit  
- Git & GitHub  

---

## 📂 Project Structure

house-price-streamlit/
│
├── app.py # Streamlit application
├── Linear.ipynb # End-to-end ML notebook
├── model.pkl # Trained ML model
├── pipeline.pkl # Preprocessing pipeline
├── data.csv # Dataset
├── requirements.txt # Dependencies
└── README.md


---

## 🚀 Deployment

The application is deployed using **Streamlit Cloud**.

### Deployment Flow
1. Train and save the model locally
2. Push project to GitHub
3. Connect the repository to Streamlit Cloud
4. Deploy using `app.py` as the entry point



---

## 📈 Key Learnings

- Importance of preprocessing pipelines
- Avoiding data leakage
- Feature selection vs user input design
- Stratified sampling for fair evaluation
- Model deployment best practices
- Converting ML notebooks into real products

---

## ⭐ Acknowledgements

- Boston Housing Dataset  
- Scikit-learn Documentation  
- Streamlit Documentation  

