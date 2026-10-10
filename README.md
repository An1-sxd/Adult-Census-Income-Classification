# Adult Census Income Classification

This project classifies whether a person's annual income is above `$50K` using the
Adult Census Income dataset. The main notebook is [`train-v2.ipynb`](./train-v2.ipynb).
It contains exploratory data analysis, preprocessing, a PyTorch neural network,
and Random Forest comparisons.

## Project goals

- Explore the Adult Census Income dataset.
- Prepare mixed numerical and categorical data for machine learning.
- Build and train a binary-classification neural network with PyTorch.
- Save and reload checkpoints selected by validation recall and F1 score.
- Compare the neural network with a Random Forest baseline and a tuned Random Forest.
- Evaluate the models with accuracy, recall, F1 score, classification reports, and
  confusion matrices.

## Repository contents

```text
.
|-- adult.csv
|-- train-v2.ipynb
|-- train-v3.ipynb
|-- train-v4.ipynb
|-- train.ipynb
`-- README.md
```

`train-v2.ipynb` is the version documented here. The other notebooks are earlier
or alternative iterations of the project.

## Dataset

The dataset contains demographic, education, employment, and financial attributes.
The target is `income`:

- `<=50K` -> `0`
- `>50K` -> `1`

The original dataset has 32,561 rows and 15 columns. After removing 24 duplicate
rows, the notebook works with 32,537 rows. The target is imbalanced:

| Class | Proportion |
|---|---:|
| `<=50K` | 75.91% |
| `>50K` | 24.09% |

Because of this imbalance, accuracy is considered together with positive-class
recall, F1 score, and the confusion matrix.

## Exploratory analysis and cleaning

The notebook:

1. Loads `adult.csv` with pandas.
2. Inspects the shape, data types, summary statistics, missing values, duplicates,
   and category distributions.
3. Removes duplicate rows.
4. Replaces `?` values in `workclass`, `occupation`, and `native.country` with
   `Unknown`.
5. Examines numerical distributions and categorical relationships with income.
6. Investigates the `capital.gain` values, including the value `99999`.
7. Maps the income labels to binary values.

## Preprocessing

Categorical features are one-hot encoded with the first category removed:

```python
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
```

The encoded data is split into training, validation, and test sets using
stratification and `random_state=42`:

```text
training set   80%
validation set 10%
test set       10%
```

Numerical columns are standardized with `StandardScaler`. The scaler is fitted
only on the training data and then applied to the validation and test data.
The resulting arrays are converted to `float32` before being wrapped in
PyTorch `TensorDataset` and `DataLoader` objects.

## PyTorch neural network

The model is a multilayer perceptron:

```text
input -> Linear(128) -> ReLU -> Dropout(0.2)
      -> Linear(64)  -> ReLU -> Dropout(0.2)
      -> Linear(1)
```

Training details:

- Batch size: `64`
- Maximum epochs: `500`
- Loss: `BCEWithLogitsLoss`
- Optimizer: SGD
- Learning rate: `0.1`
- Weight decay: `1e-4`
- Metrics: accuracy, recall, F1 score, loss, and confusion matrix

The final layer returns logits, so a separate sigmoid layer is not added.
`BCEWithLogitsLoss` combines the sigmoid operation and binary cross-entropy
loss in a numerically stable way.

During training, the notebook saves two `state_dict` checkpoints:

- `best_recall_<epoch>.pth`
- `best_f1_<epoch>.pth`

The checkpoints are selected using validation recall and validation F1 score.
The model architecture must be recreated before loading either checkpoint.

## Random Forest comparison

The notebook also trains a class-balanced `RandomForestClassifier` and searches
64 parameter combinations with five-fold cross-validation, for 320 total fits.
The grid includes:

```python
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy"],
}
```

The grid search in `train-v2.ipynb` optimizes `f1`.

## Recorded results

These results are the outputs recorded in `train-v2.ipynb`. The neural-network
checkpoint results below are from the held-out test set; the checkpoint selection
itself is based on the validation set.

| Model | Accuracy | Class-1 recall | Class-1 F1 |
|---|---:|---:|---:|
| Neural network, best-F1 checkpoint | 0.8325 | 0.7105 | 0.6715 |
| Untuned Random Forest | 0.8473 | 0.61 | 0.66 |
| Grid-search Random Forest | 0.8316 | 0.80 | 0.70 |

The neural network provides a useful PyTorch implementation and a stronger
positive-class recall than the untuned Random Forest in this run. The tuned
Random Forest has the strongest positive-class recall and F1 score among the
recorded test results, while the untuned Random Forest has the highest accuracy.
Model choice should therefore depend on whether accuracy, recall, or F1 is the
primary objective.

## How to run

### Google Colab

The notebook includes Google Drive mounting and initially reads the dataset from
a Drive path:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Update the dataset and checkpoint paths for your own Drive folders, then run
the notebook from top to bottom.

### Local Jupyter

Install the dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn torch torchmetrics jupyter
```

Open the notebook:

```bash
jupyter notebook train-v2.ipynb
```

When running locally, use the CSV in this folder:

```python
df = pd.read_csv("adult.csv")
```

## Limitations and possible improvements

- The notebook selects checkpoints with validation metrics, but the reported
  comparison still depends on one fixed random split and one training run.
- The preprocessing objects and trained model weights are not exported as a
  single reusable inference pipeline.
- The neural-network optimizer and learning rate could be tuned further.
- Early stopping could reduce unnecessary epochs and limit overfitting.
- Probability thresholds could be tuned for a chosen recall/F1 trade-off.
- Multiple random seeds and precision-recall or ROC-AUC curves would make the
  comparison more robust.

## Learning outcomes

This project practices:

- Exploratory data analysis with pandas and matplotlib.
- Handling missing categorical values and one-hot encoding.
- Standardizing numerical features without leaking test-set statistics.
- Stratified train/validation/test splitting.
- Building PyTorch datasets, data loaders, and a multilayer perceptron.
- Using logits with `BCEWithLogitsLoss`.
- Tracking classification metrics and confusion matrices.
- Saving and loading PyTorch `state_dict` checkpoints.
- Hyperparameter search with scikit-learn.
