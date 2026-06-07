# ==========================================================
# PART 1 - TOURNAMENT SETUP + DATA LOADING
# ==========================================================
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint

# TOURNAMENT CONFIGURATION
TRAIN_FOLDER = r"C:\Shoby deathless laptop folder\Nifty-Training-dataset"
MOCK_FOLDER = r"C:\Shoby deathless laptop folder\Nifty -mockexam-dataset"
FINAL_FOLDER = r"C:\Shoby deathless laptop folder\Nifty-Finalexam-dataset"

LOOKBACK_WINDOW = 90
HORIZON = 30
SEQUENCE_LENGTH = LOOKBACK_WINDOW + HORIZON
EPOCHS = 10
BATCH_SIZE = 64
TOP_K_FINALISTS = 5

# *** CACHE PATHS ***
CACHE_TRAIN_X = "cache_train_X.npy"
CACHE_TRAIN_Y = "cache_train_Y.npy"
CACHE_MOCK_X  = "cache_mock_X.npy"
CACHE_MOCK_Y  = "cache_mock_Y.npy"
CACHE_FINAL_X = "cache_final_X.npy"
CACHE_FINAL_Y = "cache_final_Y.npy"

# ARCHITECTURE CANDIDATES (10 Layers)
ARCHITECTURES = {
    "Layer_1": [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 25],
    "Layer_2": [500, 400, 300, 200, 100, 50, 25],
    "Layer_3": [500, 400, 300, 200, 100, 50, 25, 500, 400, 300, 200, 100, 50, 25],
    "Layer_4": [1000, 500, 250, 125, 60, 30, 15],
    "Layer_5": [500, 250, 100, 25, 100, 250, 500],
    "Layer_6": [500, 250, 50, 250, 50, 250, 50],
    "Layer_7": [500, 400, 300, 200, 100, 50, 25, 15, 5],
    "Layer_8": [500, 250, 100, 50, 100, 250, 500],
    "Layer_9": [512, 384, 256, 192, 128, 96, 64],
    "Layer_10": [1024, 512, 256, 128, 64, 32, 16]
}

# CREATE MODEL FROM ARCHITECTURE LIST
def build_model(layer_structure):
    model = Sequential()
    model.add(Input(shape=(LOOKBACK_WINDOW,)))
   
    for neurons in layer_structure:
        model.add(Dense(neurons, activation='relu'))
       
    model.add(Dense(HORIZON, activation='linear'))
    model.compile(optimizer='adam', loss='mae')
    return model

# DATASET LOADER
def load_dataset_folder(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    X_list = []
    y_list = []

    print(f"\nLoading Folder: {folder_path}")
    print(f"CSV Files Found: {len(csv_files)}")

    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
           
            if "Price" not in df.columns:
                continue
               
            prices = df["Price"].values.astype(np.float32)
           
            if len(prices) < SEQUENCE_LENGTH:
                continue
               
            for i in range(len(prices) - SEQUENCE_LENGTH + 1):
                window = prices[i : i + SEQUENCE_LENGTH]
                anchor_price = window[LOOKBACK_WINDOW - 1]
               
                if anchor_price <= 0:
                    continue
                   
                normalized = window / anchor_price
                X = normalized[:LOOKBACK_WINDOW]
                y = normalized[LOOKBACK_WINDOW:]
               
                X_list.append(X)
                y_list.append(y)
               
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    X_array = np.array(X_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.float32)
   
    print(f"Windows Created: {len(X_array):,}")
    return X_array, y_array

# *** CACHED LOADER — loads from .npy if already built, otherwise builds and saves ***
def load_cached(cache_x, cache_y, folder_path):
    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print(f"\n[CACHE HIT] Loading from cache: {cache_x}, {cache_y}")
        X = np.load(cache_x)
        y = np.load(cache_y)
        print(f"Windows Loaded: {len(X):,}")
        return X, y
    else:
        print(f"\n[CACHE MISS] Building cache for: {folder_path}")
        X, y = load_dataset_folder(folder_path)
        np.save(cache_x, X)
        np.save(cache_y, y)
        print(f"Cache saved: {cache_x}, {cache_y}")
        return X, y

# LOAD ALL 3 DATASETS (with cache)
print("\n==============================")
print("LOADING TRAIN DATASET")
print("==============================")
X_train, y_train = load_cached(CACHE_TRAIN_X, CACHE_TRAIN_Y, TRAIN_FOLDER)

print("\n==============================")
print("LOADING MOCK EXAM DATASET")
print("==============================")
X_mock, y_mock = load_cached(CACHE_MOCK_X, CACHE_MOCK_Y, MOCK_FOLDER)

print("\n==============================")
print("LOADING FINAL EXAM DATASET")
print("==============================")
X_final, y_final = load_cached(CACHE_FINAL_X, CACHE_FINAL_Y, FINAL_FOLDER)

# SAFETY CHECKS
if len(X_train) == 0: raise ValueError("Training dataset produced zero windows.")
if len(X_mock) == 0: raise ValueError("Mock exam dataset produced zero windows.")
if len(X_final) == 0: raise ValueError("Final exam dataset produced zero windows.")

print("\n===================================")
print("DATA LOADING COMPLETE")
print("===================================")
print(f"Train Windows : {len(X_train):,}")
print(f"Mock Windows  : {len(X_mock):,}")
print(f"Final Windows : {len(X_final):,}")
print(f"Architectures : {len(ARCHITECTURES)}")

# ==========================================================
# PART 2 - TRAIN ALL 10 ARCHITECTURES + MOCK EXAM
# ==========================================================
mock_results = []
os.makedirs("tournament_models", exist_ok=True)

# DIRECTION ACCURACY FUNCTION
def calculate_direction_accuracy(y_true, y_pred):
    actual_direction = np.sign(y_true[:, -1] - 1.0)
    predicted_direction = np.sign(y_pred[:, -1] - 1.0)
    return np.mean(actual_direction == predicted_direction) * 100.0

print("\n" + "=" * 70)
print("MOCK EXAM TOURNAMENT STARTED")
print("=" * 70)

for architecture_name, layer_structure in ARCHITECTURES.items():
    print("\n" + "-" * 70)
    print(f"Training {architecture_name}")
    print(f"Layers: {layer_structure}")
    print("-" * 70)

    model = build_model(layer_structure)
    save_path = os.path.join("tournament_models", f"{architecture_name}.keras")
   
    checkpoint = ModelCheckpoint(
        save_path,
        monitor="loss",
        save_best_only=True,
        mode="min",
        verbose=0
    )

    # *** TRAIN — verbose=0 to suppress garbled progress bar output ***
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint],
        verbose=0  # *** CHANGED: was verbose=1, caused the corrupted terminal output ***
    )

    # *** PRINT CLEAN EPOCH SUMMARY INSTEAD ***
    for epoch_idx, loss_val in enumerate(history.history["loss"], start=1):
        print(f"  Epoch {epoch_idx:>2}/{EPOCHS}  |  loss: {loss_val:.6f}")

    # LOAD BEST VERSION AND TAKE MOCK EXAM
    best_model = tf.keras.models.load_model(save_path)
    mock_predictions = best_model.predict(X_mock, verbose=0)
    mock_accuracy = calculate_direction_accuracy(y_mock, mock_predictions)
   
    print(f"\nMOCK EXAM ACCURACY: {mock_accuracy:.2f}%")
   
    mock_results.append({
        "architecture": architecture_name,
        "layers": str(layer_structure),
        "mock_accuracy": mock_accuracy,
        "model_path": save_path
    })

# CREATE MOCK LEADERBOARD AND SELECT TOP 5
mock_results = sorted(mock_results, key=lambda x: x["mock_accuracy"], reverse=True)

print("\n" + "=" * 70)
print("MOCK EXAM LEADERBOARD")
print("=" * 70)
for rank, result in enumerate(mock_results, start=1):
    print(f"{rank}. {result['architecture']} | {result['mock_accuracy']:.2f}%")

top_finalists = mock_results[:TOP_K_FINALISTS]

print("\n" + "=" * 70)
print("TOP 5 FINALISTS ADVANCING")
print("=" * 70)
for rank, finalist in enumerate(top_finalists, start=1):
    print(f"{rank}. {finalist['architecture']}")

pd.DataFrame(top_finalists).to_csv("top_5_finalists.csv", index=False)
print("\nSaved: top_5_finalists.csv - Ready for Final Exam Stage.")

# ==========================================================
# PART 3 - FINAL EXAM + CHAMPION SELECTION
# ==========================================================
finalists_df = pd.read_csv("top_5_finalists.csv")
final_exam_results = []

print("\n" + "=" * 70)
print("FINAL EXAM TOURNAMENT")
print("=" * 70)

for _, row in finalists_df.iterrows():
    architecture_name = row["architecture"]
    model_path = row["model_path"]
    mock_accuracy = row["mock_accuracy"]

    print("\n" + "-" * 70)
    print(f"Final Exam Candidate: {architecture_name}")
    print(f"Mock Accuracy: {mock_accuracy:.2f}%")
    print("-" * 70)

    model = tf.keras.models.load_model(model_path)
    final_predictions = model.predict(X_final, verbose=0)
    final_accuracy = calculate_direction_accuracy(y_final, final_predictions)

    # EXTRA REPORT CARD METRICS
    y_true_flat = y_final.flatten()
    y_pred_flat = final_predictions.flatten()
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]

    final_exam_results.append({
        "architecture": architecture_name,
        "mock_accuracy": mock_accuracy,
        "final_accuracy": final_accuracy,
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "model_path": model_path
    })
   
    print(f"Final Accuracy: {final_accuracy:.2f}%")

# SORT FINAL EXAM RESULTS AND CROWN CHAMPION
final_exam_results = sorted(final_exam_results, key=lambda x: x["final_accuracy"], reverse=True)
champion = final_exam_results[0]

print("\n" + "=" * 80)
print("FINAL EXAM LEADERBOARD")
print("=" * 80)
for rank, result in enumerate(final_exam_results, start=1):
    print(f"{rank}. {result['architecture']} | Mock={result['mock_accuracy']:.2f}% | Final={result['final_accuracy']:.2f}%")

print("\n" + "=" * 80)
print("🏆 TOURNAMENT CHAMPION REPORT CARD")
print("=" * 80)
print(f"Champion Architecture: {champion['architecture']}")
print(f"Mock Exam Accuracy:    {champion['mock_accuracy']:.2f}%")
print(f"Final Exam Accuracy:   {champion['final_accuracy']:.2f}%")
print(f"MAE:                   {champion['mae']:.6f}")
print(f"RMSE:                  {champion['rmse']:.6f}")
print(f"Correlation:           {champion['correlation']:.4f}")
print("=" * 80)

pd.DataFrame(final_exam_results).to_csv("tournament_final_results.csv", index=False)
print("\nSaved: tournament_final_results.csv")
print(f"\n🏆 WINNER: {champion['architecture']} with Final Exam Accuracy {champion['final_accuracy']:.2f}%")
print("=" * 80)
