import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


DATA_PATH = os.path.join('MP_Data_Final') 
actions = np.array(['Normal', 'Fall'])
no_sequences = 28         
sequence_length = 60     


label_map = {label:num for num, label in enumerate(actions)}
print("Label Map:", label_map)

ি
sequences, labels =[],[]


print("ডেটা লোড হচ্ছে, দয়া করে অপেক্ষা করুন...")
for action in actions:
    for sequence in range(no_sequences):
        window =[] 
        
        for frame_num in range(sequence_length):
            ছে
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(file_path)
            window.append(res) 
            
        sequences.append(window) 
        labels.append(label_map[action])

print("সব ডেটা সফলভাবে লোড হয়েছে!")


X = np.array(sequences)
y = to_categorical(labels).astype(int)


print("X এর আকার (Shape):", X.shape)
print("y এর আকার (Shape):", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("ট্রেনিং ডেটার আকার:", X_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping

print("\nনতুন স্লিম মডেল তৈরি হচ্ছে...")

model = Sequential()


model.add(LSTM(64, return_sequences=False, activation='relu', input_shape=(60, 132)))
model.add(Dropout(0.2)) 


model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))


model.add(Dense(actions.shape[0], activation='softmax'))

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()


early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=20,
    restore_best_weights=True
)


print("\nমডেল ট্রেনিং শুরু হচ্ছে...")
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=200, 
    callbacks=[early_stopping]
)

model.save('Fall_Detection_Model.h5')
print("\nমডেল সফলভাবে ট্রেইন এবং সেভ হয়েছে!")
