# Housing Exploratory Data Analysis and Prediction

This repository contains exploratory data analysis (EDA) and predictive modeling code for analyzing and predicting housing prices based on the provided dataset.
## Skills Demonstrated

1. **Data Analysis and Visualization**:
    - Utilized libraries such as Pandas, NumPy, Matplotlib, and Seaborn for data manipulation, summary statistics, and visualization.
    - Conducted thorough data exploration, including examining data shape, summary statistics, distribution of variables, and correlation analysis.
    - Visualized data distributions, relationships between variables, and identified outliers using various plots like histograms, scatter plots, box plots, and heatmaps.

2. **Data Preprocessing**:
    - Handled missing values by identifying and imputing them appropriately using techniques such as median imputation and mode imputation.
    - Detected and dealt with outliers to improve the quality of the data.
    - Performed feature engineering by transforming numerical variables, converting categorical variables into appropriate formats, and creating new features.

3. **Model Training**:
    - Prepared the data for modeling by splitting it into train and test sets.
    - Utilized pipeline and column transformer to preprocess both numerical and categorical features separately.
    - Developed predictive models using various algorithms, including Lasso Regression, Random Forest Regression, and XGBoost Regressor.
    - Evaluated model performance using cross-validation and calculated the Root Mean Squared Error (RMSE) as the evaluation metric.

4. **Model Tuning and Optimization**:
    - Fine-tuned the models using RandomizedSearchCV to search for the best hyperparameters.
    - Selected optimal hyperparameters based on the results of RandomizedSearchCV to improve model performance.

5. **Model Stacking**:
    - Implemented model stacking technique using a Stacking Regressor to combine predictions from multiple base models.
    - Evaluated the performance of the stacking regressor alongside individual base models using cross-validation and visualization techniques.

6. **Model Deployment**:
    - Saved the trained stacking regressor model using joblib for potential deployment in production environments.

## Files Description

- `housing_eda.ipynb`: Jupyter Notebook containing code for exploratory data analysis and preprocessing.
- `housing_modeling.ipynb`: Jupyter Notebook containing code for model building, tuning, stacking, and evaluation.
- `stacking_regressor.pkl`: Pickle file containing the trained stacking regressor model.

## Requirements

- Python 3.9, 3.10 or 3.11
- Jupyter Notebook
- Required Python libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost

## Installation

To run the code in this project, you need to have Python 3 installed on your system. You can install the required Python libraries using pip, a package manager for Python packages. Here's how you can install the necessary libraries:

1. **Install Python**: If you haven't already installed Python 3, you can download and install it from the [official Python website](https://www.python.org/downloads/).

2. **Install Jupyter Notebook**: Jupyter Notebook is an interactive computing environment that allows you to create and share documents containing live code, equations, visualizations, and narrative text. You can install Jupyter Notebook using pip:

    ```bash
    pip install notebook
    ```

3. **Install required libraries**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```

This command will install the following Python libraries:

- Pandas: Data manipulation and analysis library.
- NumPy: Numerical computing library.
- Matplotlib: Plotting library for creating visualizations.
- Seaborn: Data visualization library based on Matplotlib.
- Scikit-Learn: Machine learning library for building predictive models.
- XGBoost: Gradient boosting library for building efficient and scalable machine learning models.

Once you have Python and the required libraries installed, you can proceed to run the code in this project.
## Usage

1. **Exploratory Data Analysis**:
    - Open `housing_eda.ipynb` in a Jupyter environment.
    - Run the code cells to perform data exploration, visualization, and preprocessing.

2. **Model Building and Evaluation**:
    - Open `housing_modeling.ipynb` in a Jupyter environment.
    - Run the code cells to build, tune, and evaluate predictive models.

3. **Model Deployment**:
    - Load the saved stacking regressor model (`stacking_regressor.pkl`) for deployment in production environments.

Feel free to explore the code and reach out if you have any questions or feedback!