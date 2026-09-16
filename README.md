
---

## 🧹 Data Preparation

- Fixed `TotalCharges` (blank strings → 0, converted to numeric)
- Dropped `customerID` (non-predictive identifier)
- Verified no missing values and no duplicate rows after cleaning
- Encoded `Churn` as binary (Yes=1, No=0)
- Created `Tenure_Group` bins for easier dashboard filtering

---

## 📈 Exploratory Data Analysis — Key Findings

- Overall churn rate: **~26.5%**
- **Month-to-month** contracts churn far more than One/Two-year contracts
- Customers with **Fiber optic** internet churn more than DSL
- Lack of **Tech Support** and **Online Security** strongly correlates with higher churn
- Churn risk decreases sharply as **tenure** increases

---

## 🧩 Customer Segmentation (K-Means)

- Features standardized (numeric) + one-hot encoded (categorical)
- Optimal cluster count validated with **Elbow Method** and **Silhouette Score**
- Final model: **K = 3 clusters**
- Each cluster profiled by tenure, monthly charges, contract type, service adoption, and churn rate — used to build the "Customer Segments" view in the dashboard

---

## 🤖 Churn Prediction Model

Two models trained and compared:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | **0.84** |
| Random Forest | — | — | — | — | — |

- Final model: **Logistic Regression**, decision threshold tuned to **0.30** (optimized for recall — catching more at-risk customers is prioritized over precision, since the business cost of missing a churner is higher than a false alarm)
- Feature importance extracted from model coefficients to identify top churn drivers (used directly in the dashboard's "Churn Drivers" section)

---

## 📊 Power BI Dashboard

Three pages:

1. **Info** — Project overview, purpose, data source, and key questions
2. **Services** — KPI cards (Churn Rate, Customer Count, Avg Monthly Charges, Avg Tenure), churn by contract/internet service, payment method matrix, geographic distribution
3. **Executive Scorecard** — High-level summary for stakeholders

![Dashboard Info Page](assets/dashboard-info.png)
![Dashboard Services Page](assets/dashboard-services.png)

**Key DAX measure example:**
````dax
Cluster_Count = DISTINCTCOUNT(telco_churn_dashboard[Cluster])
````

---

## 🛠️ Tech Stack

**Analysis & Modeling:** Python, Pandas, Scikit-learn, Matplotlib
**Techniques:** K-Means Clustering, Logistic Regression, Random Forest, StandardScaler, OneHotEncoder
**Visualization:** Power BI (DAX measures, interactive filters, KPI cards)
**Frontend:** React, Next.js, Tailwind CSS

---

## 🚀 How to Reproduce

````bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/telco-churn-analysis.git
cd telco-churn-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook churn_analysis.ipynb

# 4. Open the .pbix file in Power BI Desktop to explore the dashboard
````

---

## 📌 Project Structure

````
telco-churn-analysis/
│
├── notebooks/
│   └── churn_analysis.ipynb
├── data/
│   └── telco_churn_dashboard.csv
├── dashboard/
│   └── Customer_Profitability_ERP_Dashboard.pbix
├── assets/
│   └── screenshots/
├── requirements.txt
└── README.md
````

---

## 🔮 Future Improvements

* Deploy the model as a real-time scoring API
* Add SHAP values for deeper model explainability
* A/B test retention offers per cluster segment

---

## 📬 Contact

**[Your Name]**
[LinkedIn](YOUR_LINK) · [GitHub](YOUR_LINK) · [Portfolio](YOUR_LINK)

````

---

بس عبيها بالـ metrics الفعلية (accuracy/precision/recall) من الأوت بوت يلي طلع معك بالخلية 32/34 وحطي روابطك الحقيقية بمكان الـ placeholders. جاهزة تنسخيها وتلزقيها بـ GitHub مباشرة.
````
