# Olist Customer Analysis and Prediction

## Author Information
**Full Name:** Muhammad Haris Bin Hadi (210963T) 

**Email Address:** 210963T@mymail.nyp.edu.sg  

---

## Project Overview

This project focuses on analyzing transactional and customer data from Olist, a Brazilian e-commerce platform, to identify repeat buyers. The goal is to provide actionable insights and develop a machine learning model to predict customer retention, enabling Olist to implement targeted marketing strategies and enhance customer loyalty.

Perform Exploratory Data Analysis (EDA) to uncover patterns in the dataset. Preprocess the data to handle missing values, aggregate relevant features, and prepare it for modeling. Engineer meaningful features that contribute to predicting repeated buyers. Build, evaluate, and optimize machine learning models to predict repeat buyers effectively. Generate actionable business insights to improve customer retention.

The submission includes an in-depth explanation of the pipeline, analysis, and the rationale behind design choices. Additionally, it demonstrates how this solution can help businesses optimize their strategies to retain loyal customers.

---

## Folder Structure

The project folder is organized as follows:

```plaintext
├── data/
│   ├── raw_dataset/
│   │   ├── olist_customers_dataset.csv
│   │   ├── olist_geolocation_dataset.csv
│   │   ├── olist_order_items_dataset.csv
│   │   ├── olist_order_payments_dataset.csv
│   │   ├── olist_order_reviews_dataset.csv
│   │   ├── olist_orders_dataset.csv
│   │   ├── olist_products_dataset.csv
│   │   ├── olist_sellers_dataset.csv
│   │   ├── product_category_name_translation.csv
│   ├── clean_dataset/
│   │   ├── cleaned_data.csv
│   ├── predictions/
│       ├── repeated_buyers_predictions.csv
├── models/
│   ├── best_model.joblib
├── src/
│   ├── dataprep.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── prediction.py
│   ├── utils.py
├── EDA.ipynb
├── EDA.pdf
├── config.yml
├── requirements.txt
├── run.sh
└── README.md
```

## Environment and Dependencies

Programming Language:
```
Python Version: 3.9+
```
OS Platform:
```
Windows 11 (64-bit)
```

Required Libraries:
All dependencies are listed in requirements.txt. Install them using the following command:

```
pip install -r requirements.txt
```

---

## Key findings from EDA

### Customer Behavior Insights
1. Monthly Order Trends:
    - Insight: A line chart was used to visualize the number of orders over time. The trends indicate peaks in specific months, reflecting seasonal buying patterns.
    - Justification: Identifying these trends is crucial for forecasting demand and understanding customer buying cycles, which supports predicting repeat buyers during peak periods.

2. Orders Per Customer:
    - Insight: A histogram generated to show the distribution of orders per customer. Most customer placed one order, with a small proportion placing multiple orders.
    - Justification: This feature is vital for identifying potential repeat buyers and understanding customer retention behavior.

---

### Delivery Performance

1. Delivery Time Distribution:
    - Insight: A histogram was created to show the delivery time in days. The mean delivery time was marked on the chart, helping assess typical delivery durations.
    - Justification: Delivery times can directly impact customer satisfaction. Faster deliveries may encourage repeat purchases, making this analysis critical for understanding buyer retention.

--- 

### Product and Payment Preferences

1. Popular Product Categories:
    - Insight: A bar chart displayed the frequency of product categories purchased. Categories such as electronics and home appliances showed higher demand.
    - Justification: Popular product categories provide insights into customer preferences, helping identify what products are more likely to attract repeat buyers.

2. Payment Types:
    - Insight: A pie chart illustrated the distribution of payment types. Credit cards dominated as the most used payment method.
    - Justification: Payment preferences indicate customer convenience and trust, which are factors in repeat purchase behavior.

---

### Feature Engineering Highlights

1. Target Variable Creation:

    - Feature: Added a binary repeated_buyers column to indicate whether a customer is a repeat buyer.
    - Justification: The target variable is the foundation of the predictive model, enabling classification of customers.

2. Delivery Duration:

    - Feature: Created a delivery_duration feature by calculating the difference between purchase and delivery dates.
    - Justification: Delivery time influences customer satisfaction, making it a valuable predictor for repeat buyers.

3. Price-to-Freight Ratio:

    - Feature: Calculated the ratio of product price to freight value.
    - Justification: This feature helps assess customer sensitivity to delivery costs relative to the product price, which can influence buying behavior.

4. Product Category Frequency:

    - Feature: Added product_category_freq, which represents the popularity of each product category.
    - Justification: This frequency measure helps identify product trends, supporting predictions about customer preferences.

5. Average Payment per Customer:

    - Feature: Calculated the average payment amount for each customer.
    - Justification: Helps understand customer spending habits and identifies high-value customers likely to make repeat purchases.

---

### Rationale for EDA and Preprocessing

- The insights from the EDA influenced feature selection, preprocessing steps, and the final machine learning pipeline design. For instance:
    - Delivery Analysis: Directly led to the creation of the delivery_duration feature.
    - Order Trends: Guided the focus on specific customer groups, such as those who purchase seasonally or frequently.
    - Product and Payment Insights: Helped identify the most relevant features for predicting repeat buyers.

---

## Key Findings from EDA

### Objectives
The Exploratory Data Analysis (EDA) phase aimed to uncover critical patterns, insights, and anomalies within the datasets. These findings informed the feature engineering process and established the foundation for the machine learning pipeline.

---

### 1. Geographic Insights
- **Analysis**: 
  - Customer distribution by states was visualized using bar plots, showing significant dominance of certain states like São Paulo in the customer base.
  - Seller distribution was similarly analyzed to identify seller hot spots.
- **Justification**: 
  - These insights are crucial for identifying regions with high customer and seller activity. High-order regions could foster repeat buyers and warrant tailored marketing or delivery strategies.

---

### 2. Customer Behavior
- **Orders Per Customer**:
  - **Insight**: The majority of customers placed only a single order. A histogram showed the distribution of orders per customer, highlighting the challenge of retaining customers as repeat buyers.
  - **Justification**: Understanding the distribution of orders is critical for identifying repeat buyers and strategizing retention efforts.
  
- **Monthly Order Trends**:
  - **Insight**: A line chart of monthly orders revealed trends over time, including potential seasonal peaks and dips.
  - **Justification**: Recognizing monthly patterns helps model customer behavior and predict periods of high or low activity.

---

### 3. Delivery Insights
- **Delivery Times**:
  - **Insight**: The distribution of delivery times was analyzed to identify average and extreme cases. 
  - **Justification**: Long delivery times could negatively affect customer satisfaction and retention, directly influencing repeat buying behavior.

- **Delivery Performance by State**:
  - **Insight**: State-level analysis showed how delivery times varied geographically.
  - **Justification**: Optimizing delivery operations in slow-performing states could improve customer satisfaction and retention.

---

### 4. Product and Payment Preferences
- **Product Categories**:
  - **Insight**: Popular product categories were identified, with specific categories having high customer interest.
  - **Justification**: This insight can be used for tailored recommendations and marketing efforts, impacting customer retention and upselling strategies.

- **Payment Methods**:
  - **Insight**: Analysis of payment methods showed customer preferences, with significant usage of a few key methods.
  - **Justification**: Understanding payment preferences can improve the customer experience and influence repeat purchases.

---

### 5. Feature Engineering Deliverables
- **Engineered Features**:
  - `repeated_buyers`: A binary column identifying customers with more than one order.
  - `delivery_duration`: Time taken to deliver an order.
  - `price_to_freight_ratio`: A metric for analyzing customer sensitivity to shipping costs.
  - `avg_payment_per_customer`: The average payment value per customer.
- **Justification**: These engineered features were designed based on EDA findings to improve the prediction of repeat buyers. Each feature contributes unique predictive value to the model.

---

### 6. Target Variable Creation
- **Insight**:
  - The binary column `repeated_buyers` was created to distinguish repeat buyers from one-time buyers.
  - **Justification**: This column serves as the target variable for the machine learning model, directly addressing the problem statement.

---

### Recommendations for Improvement
- **Graphs**:
  - Introduce a graph analyzing the relationship between product categories and payment methods.
  - Visualize customer behavior segmented by delivery times to see if delays impact repeat purchases.

- **Missing Values**:
  - Summarize missing values across all datasets in a unified table for easier tracking.

- **Geographic Insights**:
  - Incorporate further granular visualizations such as heatmaps for state-wise performance metrics.

---

## Machine Learning Pipeline

### Overview

The machine learning pipeline is designed to predict repeated buyers based on customer behavior, order details, and other transactional data. This pipeline follows a logical sequence of steps to ensure data is cleaned, features are engineered, models are trained effectively, and predictions are reliable. Below is a detailed breakdown of each step in the pipeline:

### 1. Data Loading

The pipeline begins by loading datasets as specified in the `config.yml` file. These datasets include orders, customers, payments, reviews, and more. Using a centralized configuration file allows for a consistent and scalable approach to managing dataset paths and configurations. This setup simplifies updates or changes to the datasets.

### 2. Handling Missing Values in Individual Datasets

Missing values in each dataset are addressed before any merging occurs. For example:
- Missing dates in the `orders` dataset are filled with median delivery durations or placeholders.
- Missing numerical features, such as product weight or dimensions, are imputed with median values.

This step ensures each dataset is clean before merging, reducing the risk of introducing errors and enabling accurate feature engineering later.

### 3. Aggregation by `order_id`

Datasets such as `order_items`, `payments`, and `order_reviews` are aggregated by `order_id` using predefined rules. For instance:
- `payment_value` is summed for each order.
- `review_score` is averaged for each order.

Aggregating at the `order_id` level reduces data granularity and simplifies the process of merging datasets. This step is critical for capturing meaningful summaries at the order level, such as total payment values or average review scores.

### 4. Merging Datasets

The aggregated datasets are merged sequentially based on their relationships. For example:
- The `orders` dataset is merged with `customers` and then with `order_items`, `payments`, and other datasets.

Combining all datasets into a unified structure ensures that all relevant information is available in one place. This step is vital for feature engineering and model training.

### 5. Cleaning Missing Values in the Merged Dataset

Once the datasets are merged, missing values in key columns are addressed. For instance:
- Missing delivery dates are imputed using the average delivery duration.
- Missing review scores are replaced with the mean value.

Cleaning the merged dataset ensures that features derived from these columns, such as delivery duration, are accurate and meaningful.

### 6. Aggregation by `customer_unique_id`

The data is further aggregated at the customer level to focus on customer behavior. Key steps include:
- Counting the number of orders placed by each customer (`order_count`).
- Calculating the total payments and average review scores for each customer.

This aggregation shifts the focus to customer-level insights, which are essential for predicting repeated buyers. By summarizing customer behavior, the model can better identify patterns associated with repeat purchases.

### 7. Dropping Irrelevant Columns

Columns that are not relevant to the analysis or modeling are dropped. For example:
- Columns such as `order_item_id`, `seller_zip_code_prefix`, and `product_name_length` are removed.
- This step reduces noise in the dataset and improves model performance.

By focusing on meaningful columns, the pipeline ensures that the model is trained on relevant features, enhancing its predictive power.

### 8. Feature Engineering

New features are created to capture important behavioral and transactional patterns. Examples include:
- `delivery_duration`: The number of days between order placement and delivery.
- `price_to_freight_ratio`: A measure of the value sensitivity of orders.
- `avg_payment_per_customer`: The average payment made by each customer.
- `repeated_buyers`: A binary target variable indicating whether a customer has placed more than one order.

Feature engineering enhances the dataset by adding valuable insights that directly impact the model’s ability to predict repeated buyers.

### 9. Data Splitting

The processed dataset is split into training and test sets, with stratification based on the target variable if necessary. Stratified splitting ensures balanced class distributions in both training and test datasets, which is particularly important when dealing with imbalanced datasets.

### 10. Model Training and Hyperparameter Tuning

Multiple machine learning models are trained, including Logistic Regression, Random Forest, and Gradient Boosting. Hyperparameter tuning is performed using GridSearchCV to find the optimal parameters for each model. By exploring various hyperparameter combinations, the pipeline ensures that each model is trained to achieve its best performance.

### 11. Evaluation

Models are evaluated using metrics such as:
- **Accuracy**: To measure overall correctness.
- **Matthews Correlation Coefficient (MCC)**: To evaluate performance on imbalanced datasets.
- **Classification Report**: To provide detailed metrics for each class.

These evaluation metrics help determine the best-performing model for predicting repeated buyers.

### 12. Prediction

The best model is used to make predictions on the test data. The pipeline outputs predictions and their associated probabilities, which are saved in a specified directory. Automating the prediction process ensures consistency and facilitates deployment in real-world scenarios.

---

