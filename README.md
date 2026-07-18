# Explainable Multilingual Phishing Email Detection Prototype

A final-year research prototype for detecting multilingual, code-mixed, traditional, AI-generated, and Business Email Compromise (BEC) phishing emails.

The system combines multilingual Transformer embeddings, XGBoost structured-feature probabilities, engineered email features, and a multilayer perceptron (MLP) fusion classifier. It also provides screenshot OCR, original `.eml` file analysis, URL-risk inspection, LIME and SHAP explanations, and MySQL-based feedback storage.

## Project Information

- **Project:** Phishing Email Detection for Multilingual and AI-Generated Attacks
- **Course:** CSCI 43018 – Research Project
- **Student:** Shaban A.A.
- **Student Number:** CS/2020/050
- **Supervisor:** Dr. M. I. Sabar
- **Institution:** University of Kelaniya

## Supported Classification Categories

1. Legitimate
2. Spam
3. Traditional Phishing
4. Business Email Compromise
5. AI-Generated Phishing

## Main Features

### Hybrid phishing classification

The final model combines:

```text
768 Transformer embedding features
+ 5 XGBoost class probabilities
+ 12 processed structured features
= 785 fusion features
```

The resulting feature vector is supplied to an MLP classifier that returns the final category, confidence score, risk level, and probabilities for all five classes.

### Multilingual and code-mixed processing

The application supports email content containing English, Sinhala, Tamil, Sinhala–English code mixing, and Tamil–English code mixing. The text branch uses `distilbert-base-multilingual-cased`.

### Structured feature analysis

The trained structured branch uses content-derived indicators such as:

- extracted URL count,
- URL presence,
- defanged or obfuscated URL presence,
- subject and body length,
- total character count,
- total word count,
- suspicious keyword count,
- text entropy, and
- detected language.

### Screenshot OCR

Users can upload email screenshots in PNG, JPG, JPEG, or WEBP format. The OCR module extracts editable English, Sinhala, and Tamil text before prediction.

### Original EML analysis

Users can upload an original `.eml` file to extract:

- subject and body,
- sender address,
- Reply-To,
- Return-Path,
- Message-ID,
- Received headers,
- SPF, DKIM, and DMARC results, and
- attachment metadata.

The application also reports sender-domain mismatches, authentication failures, executable files, macro-enabled documents, archive files, and double-extension filenames.

> Header and attachment indicators are currently displayed as supporting security evidence. They are not part of the trained V2 XGBoost or MLP feature vectors.

### Explainability

The prototype provides:

- **LIME:** word-level local explanations for the complete hybrid text-prediction pipeline,
- **SHAP:** structured-feature explanations for the XGBoost branch, and
- **rule-based explanations:** URL, header, authentication, and attachment indicators.

### Feedback collection

Users can confirm a prediction or submit the correct category. Feedback is stored in MySQL for review, validation, and controlled future retraining.

The application does not automatically retrain the deployed model.

## Model Results

| Model | Test accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 98.01% | 0.980 |
| Random Forest | 98.28% | 0.984 |
| CNN | 97.77% | 0.979 |
| LSTM | 87.82% | 0.886 |
| XGBoost structured model | 80.08% | 0.819 |
| Transformer–XGBoost–MLP hybrid | 95.82% | 0.948 |

Random Forest produced the highest standard benchmark accuracy. The hybrid model provides a broader multilingual, explainable, and deployable architecture by combining semantic and structured evidence.

## Technology Stack

### Machine learning and NLP

- Python
- pandas
- NumPy
- scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- XGBoost
- LIME
- SHAP

### Application and storage

- FastAPI
- Streamlit
- MySQL
- SQLAlchemy
- PyMySQL
- Tesseract OCR

## Suggested Project Structure

```text
Phishing-Email-Detection/
├── backend/
│   ├── main.py
│   ├── model_service.py
│   ├── feature_engineering.py
│   ├── explainability_service.py
│   ├── ocr_service.py
│   ├── eml_service.py
│   ├── database_service.py
│   └── feedback_service.py
├── frontend/
│   └── streamlit_app.py
├── scripts/
│   ├── preprocessing and feature-engineering scripts
│   ├── baseline training scripts
│   ├── 11_extract_transformer_embeddings_v2.py
│   ├── 12_train_xgboost_structured_v2.py
│   └── 13_train_mlp_fusion_v2.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── features_v2/
│   ├── hybrid/
│   └── robustness/
├── models/
├── reports/
│   └── results/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The actual filenames in your local project may differ slightly.

## Installation

### 1. Clone the repository

```powershell
git clone "https://github.com/Hyazeez/Phishing-Email-Detection-Prototype-for-Multilingual-and-AI-Generated-Attacks-.git"

cd "Phishing-Email-Detection-Prototype-for-Multilingual-and-AI-Generated-Attacks-"
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For Command Prompt:

```cmd
.venv\Scripts\activate
```

### 3. Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Install Tesseract OCR and ensure that the executable is available through the Windows `PATH`.

The required OCR language files are:

```text
eng.traineddata
sin.traineddata
tam.traineddata
```

### 5. Create the MySQL database

Run the following commands in MySQL. Replace the example password with a strong password.

```sql
CREATE DATABASE phishing_detection_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

CREATE USER 'phishing_user'@'localhost'
IDENTIFIED BY 'YOUR_STRONG_PASSWORD';

GRANT ALL PRIVILEGES
ON phishing_detection_db.*
TO 'phishing_user'@'localhost';

FLUSH PRIVILEGES;
```

### 6. Configure environment variables

Create a local `.env` file in the project root:

```env
DATABASE_URL=mysql+pymysql://phishing_user:YOUR_STRONG_PASSWORD@localhost:3306/phishing_detection_db?charset=utf8mb4
API_BASE_URL=http://127.0.0.1:8000
```

Do not commit `.env` to GitHub.

Provide only a safe `.env.example` file:

```env
DATABASE_URL=mysql+pymysql://username:password@localhost:3306/phishing_detection_db?charset=utf8mb4
API_BASE_URL=http://127.0.0.1:8000
```

## Running the Application

Open two PowerShell terminals from the project root.

### Terminal 1: Start FastAPI

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload
```

FastAPI should be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2: Start Streamlit

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/streamlit_app.py
```

Streamlit should be available at:

```text
http://localhost:8501
```

If the final frontend file still has its development filename, run:

```powershell
streamlit run frontend/streamlit_app_with_screenshot_ocr_and_eml.py
```

## REST API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend, model, and database status |
| POST | `/predict` | Classify an email |
| POST | `/explain` | Generate LIME and SHAP explanations |
| POST | `/extract-email-screenshot` | Extract editable text from an email screenshot |
| POST | `/extract-eml` | Parse email content, headers, authentication results, and attachments |
| POST | `/feedback` | Store a confirmation or corrected category |
| GET | `/feedback` | Retrieve stored feedback records |

### Example prediction request

```json
{
  "subject": "Account verification required",
  "body": "Your account will be suspended. Verify your details at hxxps://example[.]test/login."
}
```

Example using PowerShell:

```powershell
$body = @{
    subject = "Account verification required"
    body = "Your account will be suspended. Verify your details at hxxps://example[.]test/login."
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

## Dataset and Large-File Notice

Large generated datasets are intentionally excluded from normal Git tracking because GitHub rejects individual files larger than 100 MiB.

The following local files are examples of excluded generated artifacts:

```text
data/processed/master_dataset.csv
data/processed/master_dataset_cleaned.csv
data/features/master_features.csv
data/features/train_features.csv
data/features_v2/
data/hybrid/
*.npy
*.npz
```

These files remain on the developer's local machine and can be reproduced using the preprocessing, feature-engineering, Transformer embedding, XGBoost, and MLP scripts.

Do not upload private email content, confidential headers, database exports, passwords, `.env`, or unreviewed user feedback.

## Model Training Order

A typical V2 training workflow is:

```powershell
python scripts/11_extract_transformer_embeddings_v2.py
python scripts/12_train_xgboost_structured_v2.py
python scripts/13_train_mlp_fusion_v2.py
```

Before these scripts, run the relevant data preprocessing and V2 feature-engineering scripts to generate:

```text
data/features_v2/train_features_v2.csv
data/features_v2/validation_features_v2.csv
data/features_v2/test_features_v2.csv
```

## Security and Privacy

- Use fictional or authorised test emails.
- Do not commit real credentials or API keys.
- Do not upload confidential email content to a public repository.
- Review and anonymise feedback before using it for retraining.
- Treat model output as decision support rather than a guaranteed security verdict.
- Do not open, execute, or trust attachments merely because the model predicts a safe class.

## Prototype Boundaries

The current research prototype:

- does not perform live SPF, DKIM, or DMARC cryptographic verification,
- does not perform antivirus scanning or attachment sandboxing,
- does not query live sender or domain reputation services,
- does not continuously monitor a mailbox,
- does not automatically quarantine messages,
- does not automatically retrain from user feedback, and
- has not been evaluated as a production-scale enterprise email gateway.

## Author

**Shaban A.A.**  
B.Sc. (Hons) in Computer Science  
Faculty of Computing and Technology  
University of Kelaniya  
Student Number: **CS/2020/050**

## Academic Use

This repository was developed as a university research prototype. Use it responsibly for research, education, demonstration, and authorised security testing.
