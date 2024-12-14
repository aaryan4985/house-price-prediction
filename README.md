# **California Housing Price Prediction**

This project uses the **California Housing dataset** to predict housing prices based on various features such as location, population, income, and more. The Linear Regression model is implemented using **Python** and **scikit-learn**.

---

## **Project Overview**

The goal of this project is to:
1. Load and preprocess the **California Housing dataset**.
2. Perform Exploratory Data Analysis (EDA) to understand the data.
3. Train a **Linear Regression** model to predict housing prices.
4. Evaluate the model using metrics like **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**.
5. Visualize the relationship between actual and predicted prices.

---

## **Technologies Used**

- **Python**: Programming language
- **scikit-learn**: For model building and data handling
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Data visualization

---

## **Dataset**

The project uses the **California Housing dataset**, which is included in `sklearn.datasets`.

### Dataset Features:

| Feature Name      | Description                             |
|-------------------|-----------------------------------------|
| **MedInc**        | Median income in block group            |
| **HouseAge**      | Median house age in block group         |
| **AveRooms**      | Average number of rooms per household   |
| **AveBedrms**     | Average number of bedrooms per household|
| **Population**    | Block group population                  |
| **AveOccup**      | Average house occupancy                 |
| **Latitude**      | Latitude of block group                 |
| **Longitude**     | Longitude of block group                |

The target variable is **Price** (median house value).

---

## **Setup Instructions**

To run this project locally, follow these steps:

### **1. Clone the Repository**

```bash
git clone https://github.com/aaryan4985/house-price-prediction.git
cd house-price-prediction
```

### **2. Create a Virtual Environment (Optional)**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following content:
```
numpy
pandas
matplotlib
scikit-learn
```

### **4. Run the Project**

Run the main Python script:

```bash
python calafornia_house.py
```

---

## **Project Code**

### Key Steps:

1. **Load the California Housing dataset** using `sklearn.datasets`.
2. **Data Preprocessing**: Add target column and explore basic statistics.
3. **Train-Test Split**: Divide data into training and testing sets.
4. **Model Training**: Fit the Linear Regression model.
5. **Predictions and Evaluation**:
    - Evaluate the model with **MSE** and **RMSE**.
6. **Visualization**: Compare actual and predicted prices.

---

## **Results**

- **Evaluation Metrics**:
   - **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values.
   - **Root Mean Squared Error (RMSE):** Represents the error in the same unit as the target variable.

- **Visualization**:
   A scatter plot visualizes how well the predicted prices align with the actual prices.

---

## **Sample Output**

1. **Data Preview**:
   ```
      MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  Price
   0   8.3252      41.0  6.984127  1.023810      322.0   2.555556     37.88    -122.23  4.526
   1   8.3014      21.0  6.238137  0.971880      240.0   2.109842     37.86    -122.22  3.585
   ```

2. **Model Performance**:
   ```
   Mean Squared Error (MSE): 0.25
   Root Mean Squared Error (RMSE): 0.5
   ```

3. **Visualization**:
   ![Actual vs Predicted Prices](image_placeholder.png)

---

## **Future Improvements**

- Experiment with other regression models (e.g., Ridge, Lasso, Decision Trees).
- Perform feature scaling to improve accuracy.
- Hyperparameter tuning using GridSearchCV.
- Deploy the model as a web application using Flask or Streamlit.

---

## **License**

This project is licensed under the MIT License.

---

## **Contributing**

Contributions are welcome! If you want to improve or extend the project, fork the repository and create a pull request.

---

## **Contact**

For any queries or feedback, please contact:

- **Name**: Aaryan Pradhan
- **Email**: pradhanaaryan@gmail.com
- **GitHub**: https://github.com/aaryan4985
