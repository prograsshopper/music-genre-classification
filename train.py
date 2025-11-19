import pickle
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Parameters
DATA_PATH = "data/features_30_sec.csv"
OUTPUT_MODEL_PATH = "model.bin"
TEST_SIZE = 0.2
RANDOM_STATE = 1

XGB_PARAMS = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'multi:softprob',
    'num_class': 10,
    'eval_metric': 'mlogloss',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

NUM_BOOST_ROUND = 200


def load_and_prepare_data(data_path):
    """Load and prepare the dataset"""
    df = pd.read_csv(data_path)
    if 'filename' in df.columns:
        del df['filename']
    numerical_columns = list(df.dtypes[df.dtypes.astype(str).isin(['float64', 'int64'])].index)
    for col in numerical_columns:
        if col != 'label':
            df[col] = df[col].fillna(df[col].median())
    return df


def encode_labels(df):
    """Encode genre labels to numeric values"""
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    X = df.drop('label', axis=1)
    return X, y, label_encoder


def split_data(X, y, test_size=0.2, random_state=1):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, xgb_params, num_boost_round):
    """Train XGBoost model"""
    dtrain = xgb.DMatrix(X_train, label=y_train)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False
    )
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = y_pred_proba.argmax(axis=1)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    accuracy = (y_pred == y_test).mean()

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def main():
    """Main training pipeline"""
    df = load_and_prepare_data(DATA_PATH)
    X, y, label_encoder = encode_labels(df)
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)
    model = train_model(X_train, y_train, XGB_PARAMS, NUM_BOOST_ROUND)
    metrics = evaluate_model(model, X_test, y_test)

    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'metrics': metrics,
        'feature_names': list(X.columns),
    }
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)


if __name__ == "__main__":
    main()
