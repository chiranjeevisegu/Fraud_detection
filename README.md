This project focuses on detecting fraudulent financial transactions using both **Deep Learning** (static batch models) and **Apache Spark MLlib** (streaming, distributed processing). The primary goal is to **compare performance metrics** such as training time, GPU utilization, and model accuracy.

 📁 Dataset Used
 `/content/Bank_Transaction_Fraud_Detection.csv`

🧾 Columns Overview
- **Customer_ID, Transaction_ID, Merchant_ID** – Unique Identifiers
- **Customer Demographics:** Age, Gender, City, State, Account Type
- **Transaction Info:** Amount, Type, Date, Time, Device, Location
- **Label:** `Is_Fraud` – Binary (1 for fraud, 0 for legit)

---

## 🧹 Data Preprocessing Pipeline

### ✅ Step 1: Load & Clean Column Names
- Strip whitespace, fix malformed column names (e.g., `"Four. Age"` → `"Age"`)

### 🔐 Step 2: Drop PII Columns
- Removed: `Customer_Name`, `Customer_Contact`, `Customer_Email`

### ⚠️ Step 3: Handle Missing Values
- Dropped rows with missing target or critical numeric values
- Filled:
  - Categorical: `'Unknown'`
  - Numerical: column-wise median

### 🕒 Step 4: Extract Date & Time Features
- From `Transaction_Date` & `Transaction_Time` extracted:
  - `Transaction_Hour`, `Day`, `Month`, `DayOfWeek`, `Is_Weekend`
- Dropped raw datetime columns

### 💱 Step 5: Currency Normalization
- Created binary feature: `Currency_USD` (1 if USD, else 0)
- Dropped original currency column

### 🧠 Step 6: Categorical Encoding
- Applied `LabelEncoder` to fields like:
  - `Gender`, `Account_Type`, `Transaction_Type`, `Transaction_Device`, etc.
- Dropped high-cardinality noise: `Merchant_ID`, `Transaction_ID`, `Transaction_Description`

### 📏 Step 7: Feature Scaling
- Used `StandardScaler` on:
  - `Transaction_Amount`, `Account_Balance`, `Age`

### 📊 Step 8: Train-Test Split
- Stratified `80:20` split on `Is_Fraud` to maintain class balance
- Saved preprocessed files to:
  - `/content/X_train_clean.csv`
  - `/content/X_test_clean.csv`
  - `/content/y_train.csv`
  - `/content/y_test.csv`

---

## 🔍 Objective
Both pipelines will output:
- `Is Fraud` label (0 or 1)
- **Confidence Score** (probability of fraud)

---

## ⚙️ Tech Stack

- 🐍 Python 3.x, Pandas, NumPy, Scikit-Learn
- 🧠 Keras / PyTorch (for deep learning-LSTM)
- 🔥 Apache Spark (PySpark MLlib)
- 📈 Matplotlib / Seaborn – for visual evaluation

---

## 🔮 Future Work
- Add anomaly detection with AutoEncoders or Isolation Forest
- Deploy on GCP or AWS with real-time dashboards
- Integrate real bank APIs or anonymized simulated data


