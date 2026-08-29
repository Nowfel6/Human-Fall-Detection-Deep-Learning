import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================================
# ১. ডেটা লোড করা (আপনার বিদ্যমান ফোল্ডার থেকে)
# ==========================================
DATA_PATH = os.path.join('MP_Data_Final') 
actions = np.array(['Normal', 'Fall'])
no_sequences = 28         
sequence_length = 60      

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

print("ডেটা লোড হচ্ছে...")
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ==========================================
# ২. ৩টি ভিন্ন আর্কিটেকচার তৈরি করার ফাংশন
# ==========================================
def build_model(model_type):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, activation='relu', input_shape=(60, 132)))
    model.add(Dropout(0.2))
    
    if model_type == 'Shallow':
        # সরাসরি আউটপুট লেয়ার
        pass 
    elif model_type == 'Optimal':
        # আমাদের স্লিম মডেল (32 -> 16)
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
    elif model_type == 'Heavy':
        # অনেক বড় মডেল (128 -> 64)
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
    
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# ৩. এক্সপেরিমেন্ট শুরু (Training Loop)
# ==========================================
model_names = ['Shallow', 'Optimal', 'Heavy']
accuracies = []
best_history = None

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

for name in model_names:
    print(f"\nট্রেনিং শুরু হচ্ছে: {name} Model...")
    model = build_model(name)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
    
    # লাস্ট ভ্যালিডেশন অ্যাকুরেসি সেভ করা
    val_acc = max(history.history['val_accuracy']) * 100
    accuracies.append(round(val_acc, 2))
    
    # 'Optimal' মডেলের হিস্ট্রিটা পরে লস কার্ভ আঁকার জন্য রেখে দিচ্ছি
    if name == 'Optimal':
        best_history = history

# ==========================================
# ৪. গ্রাফ ১: Accuracy Comparison Bar Chart
# ==========================================
plt.figure(figsize=(10, 6))
display_names = ['Shallow\n(No Hidden)', 'Optimal Slim\n(32->16)', 'Heavy\n(128->64)']
bars = plt.bar(display_names, accuracies, color=['#e74c3c', '#2ecc71', '#f39c12'])

plt.ylim(0, 110)
plt.title('Effect of Dense Layers on Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Validation Accuracy (%)')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', ha='center', fontweight='bold')

plt.savefig('Comparison_Chart.png') # ছবিটি সেভ হলো
print("\n১. Comparison_Chart.png সেভ হয়েছে।")
plt.show()

# ==========================================
# ৫. গ্রাফ ২: Learning Curve (Best Model)
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(best_history.history['loss'], label='Training Loss', color='#2980b9', linewidth=2)
plt.plot(best_history.history['val_loss'], label='Validation Loss', color='#c0392b', linewidth=2)
plt.title('Optimal Model Learning Curve', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('Learning_Curve.png') # ছবিটি সেভ হলো
print("২. Learning_Curve.png সেভ হয়েছে।")
plt.show()

print("\nঅভিনন্দন! আপনার রিপোর্টের জন্য সব গ্রাফ তৈরি হয়ে গেছে।")