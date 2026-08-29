import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ১. কনফিগারেশন (আপনার ডেটার সাথে মিলিয়ে নিন)
DATA_PATH = os.path.join('MP_Data_Final') 
actions = np.array(['Normal', 'Fall'])
no_sequences = 28         # আপনি ২৮টি করে ভিডিও নিয়েছেন
sequence_length = 60      # প্রতিটি ভিডিওতে ৬০টি ফ্রেম

# ২. লেবেল ডিকশনারি তৈরি (Normal = 0, Fall = 1)
label_map = {label:num for num, label in enumerate(actions)}
print("Label Map:", label_map)

# ৩. খালি বাক্স তৈরি
sequences, labels =[],[]

# ৪. সব ফোল্ডার থেকে ডেটা কুড়িয়ে আনা (The Nested Loop)
print("ডেটা লোড হচ্ছে, দয়া করে অপেক্ষা করুন...")
for action in actions:
    for sequence in range(no_sequences):
        window =[] # এটি হলো একটি ভিডিওর প্যাকেট (৬০টি ফ্রেমের জন্য)
        
        for frame_num in range(sequence_length):
            # প্রতিটি .npy ফাইল পড়া হচ্ছে
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(file_path)
            window.append(res) # ফ্রেমটিকে প্যাকেটে ভরা হলো
            
        sequences.append(window) # পুরো প্যাকেটটিকে বড় বাক্সে ভরা হলো
        labels.append(label_map[action]) # এটি Normal নাকি Fall, তা লিখে রাখা হলো

print("সব ডেটা সফলভাবে লোড হয়েছে!")

# ৫. ডেটাকে NumPy Array-তে রূপান্তর (মডেলের জন্য প্রস্তুত করা)
X = np.array(sequences)
y = to_categorical(labels).astype(int) # 0 এবং 1 কে One-Hot Encode করা হলো

# ৬. সবচেয়ে গুরুত্বপূর্ণ টেস্ট: শেপ (Shape) চেক করা
print("X এর আকার (Shape):", X.shape)
print("y এর আকার (Shape):", y.shape)

# ৭. Train এবং Test ডেটায় ভাগ করা
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("ট্রেনিং ডেটার আকার:", X_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam # লার্নিং রেট কন্ট্রোল করার জন্য
from tensorflow.keras.callbacks import EarlyStopping

print("\nনতুন স্লিম মডেল তৈরি হচ্ছে...")

model = Sequential()

# ১. মাত্র একটি বা দুটি ছোট LSTM লেয়ার (ব্রেইন ছোট করা হলো)
# return_sequences=False করে দেওয়া হয়েছে, কারণ লেয়ার একটাই।
model.add(LSTM(64, return_sequences=False, activation='relu', input_shape=(60, 132)))
model.add(Dropout(0.2)) 

# ২. ডিসিশন মেকার (সিম্পল)
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# ৩. আউটপুট লেয়ার
model.add(Dense(actions.shape[0], activation='softmax'))

# ৪. অপটিমাইজার (লার্নিং রেট একটু কমিয়ে দিলাম যাতে সে ধীরে সুস্থে শেখে)
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# আগের মতোই Early Stopping এবং Fit করার কোড থাকবে...

# ৩. পাহারাদার বসানো (Early Stopping)
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=20, # টানা ২০ বার উন্নতি না হলে থামিয়ে দেবে
    restore_best_weights=True
)

# ৪. মডেল ট্রেনিং শুরু! (The Magic Step)
print("\nমডেল ট্রেনিং শুরু হচ্ছে...")
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=200, # ২০০ বার পড়তে বলা হলো
    callbacks=[early_stopping]
)

# ৫. মডেল সেভ করা (যাতে ভবিষ্যতে লাইভ ডিটেকশনে সরাসরি ব্যবহার করা যায়)
model.save('Fall_Detection_Model.h5')
print("\nমডেল সফলভাবে ট্রেইন এবং সেভ হয়েছে!")