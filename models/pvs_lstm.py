import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------
# 1. Load datasets
# -------------------------
df_mpu = pd.read_csv(r"D:\previous\semester 6\grad proj\project1\route-recognition\PVS\dataset_gps_mpu_left.csv")
df_labels = pd.read_csv(r"D:\previous\semester 6\grad proj\project1\route-recognition\PVS\dataset_labels.csv")

# -------------------------
# 2. Select MPU features
# -------------------------
features = [
    "acc_x_dashboard","acc_y_dashboard","acc_z_dashboard",
    "gyro_x_dashboard","gyro_y_dashboard","gyro_z_dashboard"
]
X = df_mpu[features].values

# -------------------------
# 3. Create a single road surface label
# -------------------------
def get_surface_label(row):
    if row['asphalt_road'] == 1:
        return 'asphalt'
    elif row['cobblestone_road'] == 1:
        return 'cobblestone'
    elif row['dirt_road'] == 1:
        return 'dirt'
    elif row['paved_road'] == 1:
        return 'paved'
    elif row['unpaved_road'] == 1:
        return 'unpaved'
    else:
        return 'unknown'

df_labels['road_surface'] = df_labels.apply(get_surface_label, axis=1)
labels = df_labels['road_surface'].values

# -------------------------
# 4. Encode labels
# -------------------------
encoder = LabelEncoder()
y = to_categorical(encoder.fit_transform(labels))

# -------------------------
# 5. Normalize features
# -------------------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# -------------------------
# 6. Sliding windows
# -------------------------
window_size = 10
X_seq, y_seq = [], []
for i in range(len(X) - window_size):
    X_seq.append(X[i:i+window_size])
    y_seq.append(y[i+window_size-1])  # label for last row in window

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# -------------------------
# 7. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True
)

# -------------------------
# 8. LSTM model
# -------------------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(window_size, X_seq.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(y.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# -------------------------
# 9. Train the model
# -------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2
)

# -------------------------
# 10. Evaluate
# -------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# -------------------------
# 11. Save Model & Encoder
# -------------------------
model.save("road_surface_lstm.h5")

import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Model and encoder saved successfully!")