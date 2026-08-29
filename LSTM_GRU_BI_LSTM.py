import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# ১. ডেটা লোড করা (Data Loading)
# ==========================================
DATA_PATH = os.path.join('MP_Data_Final') 
actions = np.array(['Normal', 'Fall'])
no_sequences = 28         # আপনার ভিডিওর সংখ্যা (প্রয়োজনে পরিবর্তন করে নেবেন)
sequence_length = 60      # প্রতিটি ভিডিওর ফ্রেম সংখ্যা

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [],[]

print("ডেটা লোড হচ্ছে... দয়া করে অপেক্ষা করুন।")
for action in actions:
    for sequence in range(no_sequences):
        window =[]
        for frame_num in range(sequence_length):
            # .npy ফাইলগুলো পড়া হচ্ছে
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            res = np.load(file_path)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# ডেটা স্প্লিট করা (test_size=0.1 বা 0.2 দিতে পারেন)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"ট্রেনিং ডেটার আকার: {X_train.shape}")

# ==========================================
# ২. মডেল বানানোর ডাইনামিক ফাংশন (Dynamic Model Builder)
# ==========================================
def build_model(model_type):
    model = Sequential()
    input_shape = (sequence_length, 132) # (60, 132)
    
    # মডেল টাইপ অনুযায়ী লেয়ার যোগ করা হচ্ছে
    if model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=False, activation='relu', input_shape=input_shape))
    elif model_type == 'GRU':
        model.add(GRU(64, return_sequences=False, activation='relu', input_shape=input_shape))
    elif model_type == 'Bi-LSTM':
        model.add(Bidirectional(LSTM(64, return_sequences=False, activation='relu'), input_shape=input_shape))
    
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax')) # Output layer
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# ৩. মডেল ট্রেনিং এবং তুলনা (Training & Benchmarking)
# ==========================================
model_names = ['LSTM', 'GRU', 'Bi-LSTM']
model_accuracies =[]

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

for name in model_names:
    print(f"\n[INFO] {name} মডেল ট্রেনিং শুরু হচ্ছে...")
    model = build_model(name)
    
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=150, 
        batch_size=32, 
        callbacks=[early_stopping], 
        verbose=0 # 0 দিলে স্ক্রিনে হাবিজাবি লেখা আসবে না, ক্লিন থাকবে
    )
    
    # সবচেয়ে সেরা ভ্যালিডেশন অ্যাকুরেসি বের করা
    best_val_acc = max(history.history['val_accuracy']) * 100
    model_accuracies.append(best_val_acc)
    print(f"✅ {name} মডেলের Accuracy: {best_val_acc:.2f}%")

# ==========================================
# ৪. রেজাল্ট ভিজ্যুয়ালাইজেশন (Bar Chart)
# ==========================================
print("\nবার-চার্ট তৈরি হচ্ছে...")

plt.figure(figsize=(9, 6))
colors =['#3498db', '#e67e22', '#9b59b6']
bars = plt.bar(model_names, model_accuracies, color=colors, width=0.5)

# গ্রাফের ডিজাইন
plt.ylim(60, 105) # 60 থেকে 105 পর্যন্ত স্কেল রাখা হলো
plt.title('Deep Learning Models Comparison for Fall Detection', fontsize=14, fontweight='bold')
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.xlabel('Model Architecture', fontsize=12)

# বারের ওপরে পার্সেন্টেজ লিখে দেওয়া
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# ছবি সেভ করা এবং দেখানো
plt.savefig('Architecture_Comparison_Chart.png', dpi=300, bbox_inches='tight')
print("📸 'Architecture_Comparison_Chart.png' ফোল্ডারে সেভ হয়েছে!")
plt.show()