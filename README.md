# Music Genre Classification

Datatalk Club Midterm Project - Music Genre Classification with Machine Learning

## Project Overview

This project aims to classify music genres using machine learning techniques. The system extracts audio features from music files and uses an XGBoost model to predict the genre from 10 different categories.

### Key Features
- **10 Genre Classification**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **XGBoost Model**: Fast and accurate multi-class classification
- **RESTful API**: FastAPI-based prediction service
- **Docker Support**: Easy deployment with containerization

### Project Structure
```
music-genre-classification/
├── train.py              # Model training script
├── predict.py            # FastAPI prediction service
├── notebook.ipynb        # Exploratory data analysis and experimentation
├── Dockerfile            # Docker configuration
├── pyproject.toml        # Poetry dependency management
├── data/                 # Dataset directory (after download)
│   └── features_30_sec.csv
└── model.bin            # Trained model file (after training)
```

### Model Performance
The XGBoost model is trained with the following configuration:
- **Algorithm**: XGBoost with softmax probability
- **Training/Test Split**: 80/20
- **Evaluation Metrics**: Accuracy, F1-score (macro & weighted)
- **Input Features**: 58 audio features including MFCC, spectral features, tempo, etc.

## Prerequisites

- Python 3.13 or higher
- Poetry (Python dependency manager)
- (Optional) Docker for containerized deployment

## Installation

### 1. Install Poetry

Poetry is used for dependency management in this project.

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

After installation, make sure Poetry is in your PATH. You may need to restart your terminal.

### 2. Install Python Dependencies

Install all required packages using Poetry:

```bash
poetry install
```

This will install the following main packages:
- xgboost (3.1.1+) - Machine learning model
- pandas (2.3.3+) - Data manipulation
- scikit-learn (1.7.2+) - ML utilities and metrics
- fastapi (0.121.3+) - Web framework
- uvicorn (0.38.0+) - ASGI server
- jupyter (1.1.1+) - Notebook environment
- matplotlib, seaborn - Data visualization

### 3. Download Dataset

The GTZAN Dataset contains audio features extracted from 1000 audio tracks (100 tracks per genre, 30 seconds each).

**Dataset Link**: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

#### Option A: Download via Kaggle API (Recommended)

```bash
# Download and extract dataset
curl -L -o gtzan-dataset-music-genre-classification.zip \
https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification

# Unzip data
unzip gtzan-dataset-music-genre-classification.zip

# Remove zip file (optional)
rm gtzan-dataset-music-genre-classification.zip

# Rename directory to lowercase
mv Data data
```

#### Option B: Manual Download

1. Visit the [dataset page](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
2. Download the dataset manually
3. Extract it to the project root directory
4. Rename the `Data` folder to `data`

After downloading, verify that `data/features_30_sec.csv` exists.

## How to Run

### Step 1: Train the Model

Train the XGBoost model using the dataset:

```bash
poetry run python train.py
```

This will:
- Load the dataset from `data/features_30_sec.csv`
- Preprocess features (handle missing values, encode labels)
- Split data into train/test sets (80/20)
- Train XGBoost model with 200 boosting rounds
- Evaluate model performance (accuracy, F1-score)
- Save the trained model to `model.bin`

**Expected Output**: Model training logs and evaluation metrics will be displayed.

### Step 2: Run the API Server

After training, you can run the prediction API server.

#### Option A: Run with Uvicorn (Development)

```bash
poetry run uvicorn predict:app --host 0.0.0.0 --port 8000
```

#### Option B: Run with Docker (Production)

Build the Docker image:
```bash
docker build -t music-genre-classifier .
```

Run the container:
```bash
docker run -p 8000:8000 music-genre-classifier
```

### Step 3: Test the API

Once the server is running, you can access:

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Root Endpoint**: http://localhost:8000/
  - Returns model metrics and available genres

#### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "length": 66149,
      "chroma_stft_mean": 0.335,
      "chroma_stft_var": 0.091,
      "rms_mean": 0.130,
      "rms_var": 0.003,
      "spectral_centroid_mean": 1773.065,
      "spectral_centroid_var": 167541.63,
      "tempo": 149.976,
      "mfcc1_mean": -193.81,
      "mfcc1_var": 2283.52
      ... (other features)
    }
  }'
```

**Expected Response**:
```json
{
  "genre": "rock",
  "confidence": 0.85,
  "probabilities": {
    "blues": 0.02,
    "classical": 0.01,
    "country": 0.03,
    "disco": 0.02,
    "hiphop": 0.01,
    "jazz": 0.02,
    "metal": 0.04,
    "pop": 0.00,
    "reggae": 0.00,
    "rock": 0.85
  }
}
```

## Development

### Jupyter Notebook

Explore the data and experiment with models using the provided notebook:

```bash
poetry run jupyter notebook notebook.ipynb
```

### Project Workflow

1. **Data Exploration**: Analyze dataset in `notebook.ipynb`
2. **Model Training**: Run `train.py` to train and save the model
3. **API Service**: Run `predict.py` to serve predictions
4. **Deployment**: Use Docker for production deployment

## Troubleshooting

### Issue: "No module named 'xgboost'"
**Solution**: Run `poetry install` to install all dependencies.

### Issue: "FileNotFoundError: data/features_30_sec.csv"
**Solution**: Make sure you've downloaded and extracted the dataset to the `data/` directory.

### Issue: "FileNotFoundError: model.bin"
**Solution**: Train the model first by running `poetry run python train.py`.

### Issue: Port 8000 already in use
**Solution**: Use a different port: `poetry run uvicorn predict:app --host 0.0.0.0 --port 8001`

