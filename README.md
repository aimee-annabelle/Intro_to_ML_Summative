# Drug Demand Classification Model

## Overview

Medicine shortage is a real problem for remote clinics in developing regions, without proper data systems to track what's needed when, these clinics often find themselves scrambling to restock essential medications after they've already run out. Most places still manage their inventory the old-fashioned way – with pen and paper, which means they're always struggling to catch up instead of staying ahead of demand. 

This project implements a neural network-based classification system to predict drug demand levels (Low, Medium, High) in remote health clinics. The system aims to optimize supply chain efficiency by categorizing demand patterns to help inventory management decisions.


**Dataset source:** [Kaggle: Pharma sales data](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data/data?select=salesweekly.csv)

Features:
- Date, total demand
- Demand category(low/medium/high)

## Findings




### Results table 

| Training instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1 score | Recall | Precision |
| ----------------- | --------- | ----------- | ------ | -------------- | ------ | ------------- | -------- | -------- | ------ | --------- |
| 1 | Adam (default) | L2 (0.001) | 50 | No | 4 | 0.001 | 0.8305 | 0.8259 | 0.8316 | 0.8295 |
| **2** | **Adam** | **L2 (0.010)** | **22** | **Yes** | **4** | **0.010** | **0.8814** | **0.8820** | **0.8825** | **0.8822** |
| 3 | RMSprop | L2 (0.005) | 75 | No | 4 | 0.005 | 0.7797 | 0.7606 | 0.7816 | 0.7895 |
| 4 | SGD | L2 (0.010) | 100 | Yes | 4 | 0.010 | 0.7627 | 0.7516 | 0.7632 | 0.7703 |
| 5 | Adam | L2 (0.100) | 50 | No | 4 | 0.100 | 0.3390 | 0.1688 | 0.3333 | 0.1130 |

## Best combination discussion


**Instance 2** achieved the highest performance with the following configuration:
- **Optimizer**: Adam
- **Learning Rate**: 0.010
- **Regularization**: L2 with λ = 0.010
- **Dropout**: 0.4
- **Early Stopping**: Yes (stopped at epoch 22)
- **Architecture**: 4 layers (128→64→32→3 neurons)

### Optimization Techniques Analysis

1. **Learning Rate Impact**: 
   - Moderate increase (0.001 → 0.010) improved performance significantly
   - Excessive increase (0.100) caused severe performance degradation and instability

2. **Early Stopping Benefits**:
   - Prevented overfitting in Instance 2 (stopped at epoch 22 vs. planned 100)
   - Instance 4 with SGD + Early Stopping still underperformed due to optimizer limitations

3. **Regularization Effects**:
   - Moderate L2 regularization (0.010) worked optimally
   - Too strong regularization (0.100) severely hindered learning

4. **Optimizer Comparison**:
   - **Adam**: Best overall performance with adaptive learning rates
   - **RMSprop**: Moderate performance, slower convergence
   - **SGD**: Poor performance despite early stopping, needs momentum

## ML Algorithm vs Neural Network Comparison

**Best perfoming: Neural Network (Instance 2)**

**Neural Network Advantages:**
- **Highest Accuracy**: 88.14% vs. 86.44% for classical ML
- **Better Generalization**: Superior F1-score (0.8820) indicates balanced performance
- **Feature Learning**: Automatically learns complex feature interactions
- **Scalability**: Can handle larger datasets and more complex patterns

**Classical ML Performance:**
- **Logistic Regression** achieved competitive results (86.44% accuracy)
  - **Hyperparameters**: C=10, solver='lbfgs', max_iter=1000
  - **Strengths**: Fast training, interpretable coefficients
  - **Limitations**: Linear decision boundaries, limited feature interaction modeling

- **XGBoost** performed similarly to Logistic Regression (86.44% accuracy)
  - **Hyperparameters**: n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8
  - **Strengths**: Handles non-linear relationships, feature importance
  - **Limitations**: Required extensive hyperparameter tuning, prone to overfitting

### Why Neural Networks Excelled

1. **Non-linear Pattern Recognition**: Drug demand involves complex temporal and seasonal patterns
2. **Feature Interaction Learning**: Automatically discovered relationships between different drug categories
3. **Regularization Flexibility**: Dropout and L2 regularization provided better generalization
4. **Optimization Adaptability**: Adam optimizer's adaptive learning rates suited the problem complexity


**Video Presentation:** [Link](https://drive.google.com/file/d/1ILA-snQLAuNpfn-11ghyVIL5fPx62N1M/view?usp=sharing)

## Usage Instructions
### Prerequisites

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Step-by-Step Execution

1. **Data Preparation**:
   ```python
   # Replace sample data generation with your actual dataset
   df = load_and_preprocess_data('your_drug_data.csv')
   ```

2. **Run All Cells Sequentially**:
   - Execute each cell in order
   - Training takes approximately 10-15 minutes on standard hardware
   - Monitor outputs for any errors or warnings

3. **Model Training**:
   - All models train automatically
   - Best model saves as 'best_drug_demand_model.keras' in the saved_models folder
   - Results tables display after training completion

### Loading and Using the Best Saved Model

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the best performing model
best_model = load_model('best_drug_demand_model.keras')
# Make predictions on new data
def predict_demand_category(model, new_data_scaled):
    """
    Predict demand categories for new data
    
    Args:
        model: Loaded Keras model
        new_data_scaled: Preprocessed and scaled feature data
    
    Returns:
        predictions: Array of predicted classes (0=Low, 1=Medium, 2=High)
        probabilities: Array of prediction probabilities
    """
    probabilities = model.predict(new_data_scaled)
    predictions = np.argmax(probabilities, axis=1)
    
    # Convert to readable labels
    labels = {0: 'Low Demand', 1: 'Medium Demand', 2: 'High Demand'}
    predicted_labels = [labels[pred] for pred in predictions]
    
    return predictions, predicted_labels, probabilities

# Example usage
# new_predictions, labels, probs = predict_demand_category(best_model, X_new_scaled)
```
## Conclusion

Overall, the optimized neural network (Instance 2) successfully outperformed classical machine learning approaches by 2% in all metrics. The combination of Adam optimizer with moderate learning rate increase, L2 regularization, and early stopping proved most effective for this drug demand classification task. This system provides actionable insights for health clinic supply chain optimization while maintaining high accuracy and reliability.