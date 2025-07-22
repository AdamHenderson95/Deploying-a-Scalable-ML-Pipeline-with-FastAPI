# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary classification model trained to predict whether an individual's income exceeds $50K/year using features from the UCI Adult Census dataset. It is based on a RandomForestClassifier from scikit-learn and trained with standard preprocessing steps including one-hot encoding of categorical features.

The model is saved and deployed using a reproducible pipeline implemented with Python and FastAPI.
## Intended Use
This model is intended for educational and demonstration purposes, specifically for showcasing a scalable ML deployment pipeline with FastAPI and W&B. It is not intended for real-world decision-making as it has not been audited for fairness, security, or robustness in production environments.
## Training Data
The model was trained using the `census.csv` dataset (derived from the UCI Adult dataset), which contains demographic information such as age, education, workclass, marital status, and income. The target variable is `salary`, binarized as `>50K` or `<=50K`.

The dataset was split into training and test sets using a stratified train-test split with a default 80/20 ratio. The training set includes both numerical and categorical features. Categorical features were encoded using a `OneHotEncoder`, and the labels were binarized using a `LabelBinarizer`.
## Evaluation Data
The test set (20% split from the original dataset) was used for evaluation. Model metrics were also computed for slices of the test data based on individual categorical feature values to evaluate fairness and performance consistency across subgroups.

## Metrics
- **Precision**
- **Recall**
- **F1-score**

Overall model performance:
- **Precision**: 0.7419
- **Recall**: 0.6384
- **F1-score**: 0.6863

### Slice-Based Metrics
To evaluate performance fairness, metrics were computed on slices of the data defined by values in categorical features like `education`, `workclass`, and `native-country`. Notable findings:

- `education: HS-grad → F1: 0.69`
- `education: 7th-8th → F1: 0.00` _(very low, likely underrepresented)_
- `relationship: Own-child → F1: 0.30`
- `native-country: Greece → F1: 0.00` _(likely due to very few examples)_
## Ethical Considerations
This model may exhibit bias against underrepresented groups due to class imbalance in the training data. Certain demographic groups, such as individuals with lower education levels or from less-represented countries, are not predicted accurately by the model, as shown in slice performance analysis.

It is important to note:
- The dataset reflects historical bias present in U.S. Census data.
- Poor performance on some slices could reinforce unfair stereotypes if used in practice.
- No mitigation strategies (such as reweighting, adversarial debiasing, or fair sampling) were applied.
## Caveats and Recommendations
Consider collecting more data or balancing underrepresented groups before deployment.
- Future improvements could involve:
  - Hyperparameter tuning
  - Model calibration
  - Debiasing techniques for fairness
  - Threshold optimization for specific subgroups
