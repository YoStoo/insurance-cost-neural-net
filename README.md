# Predicting Medical Insurance Costs with a Neural Network

A Keras neural network that estimates a person's yearly medical insurance bill from their age, sex, BMI, number of children, smoking status, and region.

Built during the TripleTen AI/ML bootcamp, Sprint 12 (Neural Networks). The project was reviewed and accepted on November 10, 2026.

The full write-up, charts, and saved outputs are in [`insurance_nn.ipynb`](insurance_nn.ipynb).

## The data

The dataset is a public list of **1,338** people and their yearly medical charges, loaded from [TripleTen's copy of the insurance data](https://raw.githubusercontent.com/TripleTen-DS/Dataset/refs/heads/main/insurance.csv). There were no missing values. I dropped 1 duplicate row, leaving **1,337** people.

The columns are `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and the target, `charges`.

Smoking is the dominant pattern in the raw data: smokers have much higher bills at every age.

## Approach

1. **Look at the data.** Check types, missing values, duplicates, and how charges are spread out. A few people have very large bills, so the target is lopsided.
2. **Clean and encode.** Turn `sex` and `smoker` into 0/1, and turn `region` into one-hot columns, for 8 input features.
3. **Hide some people.** Split 80/20 (1,069 train / 268 test) *before* scaling, and stratify on smoker so both sets have the same share of smokers. Fit a `StandardScaler` on the train set only for `age`, `bmi`, and `children`, so no information from the test set leaks into training.
4. **Train the neural net.** Train on `log1p(charges)` and convert predictions back to dollars with `expm1`, because the log version is easier for the network to learn. The model is a Keras Sequential net with Dense ReLU hidden layers and 1 output, trained with Adam, MSE loss, and MAE as the metric. Training uses batch size 32, up to 500 epochs, a 20% validation split, and early stopping (patience 20) so the run keeps the best weights and quits once validation loss stops improving.
5. **Grade it.** Compare three architectures on validation MAE, then score the winner on the held-out test set, and compare it with a lazy baseline that guesses the median bill for everyone.

I picked the best model by **validation MAE**, not test MAE, so the test set stays a fair final check.

## Results

These numbers are from the TripleTen run, with TensorFlow 2.7.0, and are saved in the notebook.

| config | params | best epoch | train MAE | val MAE |
| --- | --- | --- | --- | --- |
| deeper + dropout (64-32, 0.2) | 2,689 | 181 | `$2,608` | **`$3,319`** |
| deeper (64-32) | 2,689 | 113 | `$2,511` | `$3,494` |
| baseline (32) | 321 | 207 | `$2,566` | `$3,585` |

- **Best model:** 64 → 32 → 1 with 0.2 dropout (2,689 parameters).
- **Test set:** MAE **`$2,359`**, R² **0.863**, median error `$823`.
- **Train set:** MAE `$2,749`, R² 0.800.
- **Baseline:** always guessing the median bill gives a test MAE of `$8,141`, so the model cuts the error by about **71%**.
- **Fit:** test MAE is 14% *lower* than train MAE, and the validation loss flattens instead of rising, so the model is not overfitting. It isn't underfitting either, since it beats the baseline by a wide margin.
- **Where it struggles:** smokers have a mean error of `$6,352`, versus `$1,328` for non-smokers. Smokers' bills vary much more, and the six input columns don't fully explain that variation.

### What matters most

Permutation importance (how much test MAE rises when that column is shuffled):

| feature | increase in test MAE |
| --- | --- |
| smoker | `+$6,542` |
| age | `+$3,029` |
| bmi | `+$1,092` |
| children | `+$334` |
| sex | `+$67` |
| region | `-$41` |

Smoker wins easily, then age, then BMI. Children, sex, and region barely matter. Dropout helped a little, and the small baseline was close behind, so this data doesn't need a big network. Most of the gain came from preparing the data (the log target, scaling, and encoding), not from making the network bigger.

## What I'd do next

- Add a smoker × BMI feature, since smokers with a high BMI tend to have much larger bills.
- Try a Huber loss, so the most expensive smokers aren't under-predicted after the log transform.
- Use k-fold cross-validation instead of one validation split.
- Compare against a simpler model, such as gradient boosting.

## How to run

The notebook loads the data from the URL above, so you don't need to download a file first.

```bash
git clone https://github.com/YoStoo/insurance-cost-neural-net.git
cd insurance-cost-neural-net
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook insurance_nn.ipynb
```

Then run the cells from top to bottom. The committed notebook already has the outputs and charts from the accepted TripleTen run, so you can read the results without rerunning it.

## A note about TensorFlow versions

The accepted notebook ran on TripleTen's platform, which had **TensorFlow 2.7.0**. An early version of the setup cell called `tf.config.experimental.enable_op_determinism()`, which makes TensorFlow's math repeatable, but that setting does not exist in 2.7. The notebook now turns it on only if the installed TensorFlow has it:

```python
if hasattr(tf.config.experimental, 'enable_op_determinism'):
    tf.config.experimental.enable_op_determinism()
```

That guard is why the same notebook runs on TripleTen's 2.7 install and on a newer local install. `requirements.txt` pins a current TensorFlow 2.15 set that works on Python 3.10 or 3.11. If you rerun the notebook on a newer TensorFlow, the graphs will be the same, but the exact dollar amounts may shift a little.

## Tech stack

Python · TensorFlow 2.7 (on TripleTen) / 2.15 (local pins) · Keras · pandas · NumPy · scikit-learn · Matplotlib · Jupyter Notebook

## Project structure

```
insurance-cost-neural-net/
├── insurance_nn.ipynb   # Full notebook with saved outputs and charts
├── requirements.txt     # Pins for a local rerun on Python 3.10 or 3.11
└── README.md
```

The project followed the TripleTen Sprint 12 brief: load and preprocess the data, build and train a neural network, evaluate it, plot training/validation loss and predicted vs actual values, and write a short summary of the approach, results, and reflection.
