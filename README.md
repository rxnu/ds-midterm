# Data Science Midterm Project

## Project/Goals
The overall goal of this midterm project was to create an ML model which predicted the sale prices of houses through real-world housing data. The dataset provided came in the form of multiple JSON filed stored in a directory. 

## Process
### STEP 1: Exploratory Data Analysis (EDA) 
We loaded housing sales data from multiple JSON files, merged them into one DataFrame, and cleaned the data by handling missing values and fixing data types. We used visualizations like histograms and correlation heatmaps to explore relationships between features and the target variable (price). Key variables like living area and number of bedrooms were positively correlated with price. We saved cleaned training and testing datasets as CSVs in a processed/ folder.

### STEP 2: Model Selection 
We compared five baseline models: Linear Regression, Support Vector Regressor (SVR), Random Forest, Gradient-Boosted Trees (HistGB), and K-Nearest Neighbors. Evaluation was based on RMSE, MAE, and R², all calculated using the log-transformed sale price. Random Forest delivered the best performance, with an RMSE of 0.022 and R² of 0.999. We also explored feature selection using Lasso and RFE. Lasso selected nearly all features, while RFE reduced the set to 10 features without reducing accuracy, showing that a leaner model can match full-feature performance in both Random Forest and HistGB.

### Step 3: Tuning and Pipeline 
we focused on building a full prediction pipeline rather than grid search or hyperparameter tuning. We loaded raw JSON housing data, applied consistent preprocessing (including missing value imputation and scaling), and loaded our pre-trained Random Forest model. A Pipeline object was built to encapsulate preprocessing and prediction, ensuring no data leakage. We also saved the final pipeline using joblib for future inference on unseen data in the same JSON format. Though tuning wasn’t performed, this notebook successfully demonstrated reproducible prediction using the selected best-performing model.

## Results
Our final Random Forest model performed exceptionally well on the log-transformed housing price data, achieving an RMSE of 0.022, MAE of 0.007, and R² of 0.999. Five models were evaluated without hyperparameter tuning, including SVR, Gradient Boosted Trees, and Linear Regression. Feature selection using RFE showed that reducing the feature set to 10 did not significantly reduce performance, supporting model simplification. The final pipeline included median imputation and scaling, and it was saved for consistent predictions on new data. Overall, we achieved strong performance with a simple, interpretable, and efficient modeling workflow.

## Challenges 
One key challenge was that the predicted values appeared significantly lower than typical housing prices, which made it difficult to interpret results in a real-world context. This was due to working with believed log-transformed targets for consistency and model performance. During the EDA phase, we also encountered missing data across multiple numerical features, requiring  imputation to avoid bias. Additionally, inconsistencies in the structure of nested JSON records created difficulties in engineering features. Ensuring the data was clean and usable before modeling took a considerable amount of time.

## Future Goals
If we had more time, one of our main priorities would be reversing the logarithmic transformation before evaluating predictions, so we could better compare model outputs to real-world housing prices. This would make the results easier to interpret and more useful in practice. We’d also focus on improving the pipeline’s flexibility (i.e, automating raw data cleaning, preprocessing, and exporting predictions). Exploring more advanced feature engineering (like grouping year built or better geolocation data) could strengthen model accuracy. Lastly, we’d work on model explainability by testing tools to clearly show which features drive pricing predictions.
