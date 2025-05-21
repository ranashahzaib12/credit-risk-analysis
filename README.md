# 💳 Credit Risk Prediction App

## Project Overview

This project is an interactive **Credit Risk Prediction App** built using **Streamlit**. The app predicts the probability that a credit applicant will experience serious delinquency within 2 years. It leverages machine learning with a Gradient Boosting model, enhanced with feature engineering and preprocessing, to provide accurate and interpretable credit risk predictions.

---

## What I Did

- **Data Preparation & Feature Engineering:**  
  Created new meaningful features such as `AgeGroup` and `IncomeGroup` from raw inputs to improve the predictive power of the model.

- **Model Training and Selection:**  
  Trained multiple classifiers (Random Forest, Gradient Boosting, XGBoost) on SMOTE-balanced data, evaluated them using multiple metrics (ROC AUC, F1, precision, recall), and selected the best model (Gradient Boosting) for production use.

- **Preprocessing Pipeline:**  
  Built and saved a preprocessing pipeline for feature scaling and encoding, ensuring consistent input transformation during prediction.

- **Streamlit App Development:**  
  Developed a clean and user-friendly Streamlit app that accepts user inputs, applies feature engineering and preprocessing, and displays real-time risk predictions with styled risk messages.

- **Model & Preprocessor Serialization:**  
  Saved the trained model and preprocessing pipeline as `.pkl` files for efficient loading in the app.

---

## What I Achieved

- Developed a production-ready credit risk prediction tool that can be used by financial institutions to evaluate applicant risk quickly and reliably.
- Implemented comprehensive input validation and interactive UI elements for better user experience.
- Demonstrated best practices in model evaluation by considering multiple metrics, not just accuracy.
- Created a modular and maintainable codebase separating feature engineering, modeling, and UI logic.
- Enabled easy deployment through Streamlit, facilitating rapid prototyping and sharing.

---

## How to Use

1. Clone the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the app using `streamlit run app.py`.
4. Enter applicant details in the input fields.
5. Click **Predict Risk** to see the delinquency probability and risk classification.

---

## Project Files

- `app.py`: Streamlit app source code with feature engineering and UI.
- `best_model.pkl`: Pretrained Gradient Boosting model.
- `preprocessor.pkl`: Feature preprocessing pipeline.
- `requirements.txt`: Python dependencies.
- `README.md`: This project documentation.

---

## Dependencies (requirements.txt)

"# credit-risk-analysis" 
