# Dataset

This project uses the **Kaggle Credit Card Fraud Detection dataset**
(European cardholders, September 2013). It is ~150 MB, so it is not bundled
with this project — download it yourself:

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Sign in to Kaggle (free account).
3. Click **Download** (or use the Kaggle CLI — see below).
4. Unzip it, and place `creditcard.csv` directly inside this `data/` folder,
   so the path is:

   ```text
   Fraud_Detection_ML/data/creditcard.csv
   ```

### Alternative: Kaggle CLI

```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

### Expected columns

`Time, V1, V2, ..., V28, Amount, Class`

- `Time` — seconds elapsed between this transaction and the first transaction in the dataset
- `V1`–`V28` — PCA-transformed features (anonymized for confidentiality by the dataset owners)
- `Amount` — transaction amount
- `Class` — target label: `0` = legitimate, `1` = fraud

The dataset contains 284,807 transactions, of which only 492 (~0.17%) are
fraudulent — this is the class imbalance the project is built around.
