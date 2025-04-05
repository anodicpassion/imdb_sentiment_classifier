from data.load_data import load_imdb_data
from models.lstm_model import build_lstm_model
from tensorflow.keras.callbacks import EarlyStopping

# Load data
x_train, y_train, x_test, y_test = load_imdb_data()

# Build and compile model
model = build_lstm_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
model.fit(x_train, y_train,
          epochs=10,
          batch_size=64,
          validation_split=0.2,
          callbacks=[early_stop],
          )

# Save model
model.save("saved_model/model.h5")
