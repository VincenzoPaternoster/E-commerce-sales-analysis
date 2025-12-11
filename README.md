# 📌 E-commerce sales analysis
A mid-size European e-commerce wants to improve sales performance and customer retention.  

This project analyses an AI-generated e-commerce dataset from 2024 with the goal of practising the complete data analysis workflow — from data cleaning to final business insights.
Throughout the project, I applied:

- SQL for data cleaning, transformation, and exploration
- Power BI and Tableau for dashboard creation and business storytelling
- Python for RFM segmentation and machine learning-based clustering

The project provided an opportunity to explore customer behaviour patterns, uncover key performance drivers, and turn data into useful insights.

---

## 📂 Repository Structure
```
E-commerce-sales-analysis/
│── data/
      ├── Original data
      ├── Edited data
│── images/
      ├── Distributions
      ├── Clustering
      ├── RFM Insights
│── tableau/
      ├── E-commerce_sales_analysis.twbx
      ├── Link to Tableau Public
│── powerbi/
      ├── E-commerce_sales_analysis.pbi
      ├── E-commerce_sales_analysis.pdf
│── sql/               
│     ├── cleaning.sql
│     ├── modeling.sql
│── python/
      │──RFM Segmentation of an E-commerce customers.ipynb.
│── README.md
```

## 🎯 Project objectives
- 1 Improve data quality across key entities
- 2 Analyze revenue trends and product/category performance
- 3 Understand customer behavior across countries and channels
- 4 Build executive dashboards for decision making 

---

## 🗂️ Dataset
**Source:** AI-generated (ChatGPT)
**Periodo analizzato:** 2024  
**Dimensione del dataset:** 350 customers-3173 orders-220 products

### 📌 Key variables
| Variables | Description |
|----------|-------------|
| Order_id | Orders identification number |
| Customer_id | Customer identification number |
| Product_id | Product identification number |
| Amount | Total revenue per order |
| Quantity | Quantity of products per order |
| Price | Price per product |


---

## 🧹 Data Cleaning
Key operations performed:
- Handling missing values
- Format correction (dates, numbers, etc.)
- Duplicate removal
- Handling negative and inconsistent values
- Feature engineering (e.g. obtaining season from date)
---

## 📊 Methodology / Analysis:
- Used techniques (RFM segmentation,k-means clustering,Yeo-Johnson transformation,Standardization)
- Main libraries (Pandas,Numpy,Sklearn,Scipy,Matplotlib,Seaborn)

  Yeo–Johnson transformation applied to the RFM features to reduce skewness and stabilize variance.
  The original distributions of Frequency, Recency and Monetary were highly skewed, and K-Means is sensitive to extreme values.
  Yeo–Johnson was chosen because, unlike Box–Cox or log transformation, it can handle zero values (present in Recency).
  After the transformation, the features were standardized before fitting the clustering model.
---

## 🔍 Key results
- Insight 1: Revenue in 2024 presents high volatility in all the quarters.
             There are several peaks but the overall direction is slightly upward, indicating a soft improvement in sales throughout the year.
- Insight 2: There are products that need to be eliminated, recalibrated and promoted
- Insight 3: The best product in terms of revenue is 212
- Insight 4: The most used channel is the web, followed by mobile
- Insight 5: E-commerce customers can be divided into five clusters. Actionable commercial strategies have been provided for each cluster

(Inserire eventuali grafici nella cartella /images e richiamarli con markdown)

Esempio:
![Esempio grafico](images/example.png)

---

## 🧠 Conclusions
- This project allowed me to manage a complete end-to-end data analysis workflow, from data preparation to modelling and visualization.
  I applied different analytical techniques — including RFM segmentation, Yeo–Johnson transformation and K-Means clustering — and learned
  how these methods can be combined to extract actionable insights about customer behaviour.

- The project also help me to improve my ability to structure work efficiently and select the appropriate tools (SQL, Python, Power BI, Tableau) based on the business questions.
  Overall, this experience gave me a base for future customer analytics and segmentation projects.
---

## 🛠️ Tools
- Python (pandas, numpy, matplotlib, seaborn…)
- Google Colab
- SQL Server Management Studio
- PowerBI
- Tableau
- Obsidian

---

## 📬 Contacts

- **Vincenzo Paternoster**
- Email: vincenzopaternoster99@gmail.com
- LinkedIn: www.linkedin.com/in/vincenzo-paternoster
