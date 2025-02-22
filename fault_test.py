# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datetime import datetime
from tensorflow.keras.models import load_model


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#v1 pre-code

df = pd.read_csv('data/detect_dataset.csv')

df.head()

df.shape

df.info()

df.describe()

import pandas as pd

start_time = pd.to_datetime("2015-01-01 00:00:00")

# Number of rows in the dataset
num_rows = len(df)  # Assuming 'df' is your dataset

# Generate a range of timestamps (assuming 1-second interval)
timestamps = pd.date_range(start=start_time, periods=num_rows, freq='H')  # 'S' for second, 'T' for minutes and 'H' for hours

# Add the generated timestamps to the dataframe
df['Timestamp'] = timestamps

# Check the first few rows to ensure timestamps are correctly added
print(df[['Timestamp', 'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].head())


#Data Cleaning

df.isnull().sum()

df.drop(['Unnamed: 7','Unnamed: 8'],axis=1,inplace = True)

df.duplicated().sum()

#EDA

df['Output (S)'].unique()

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.bar(x=df['Output (S)'].unique(),height = df['Output (S)'].value_counts())
ax1.set_xticks(ticks=[0,1])

ax2.pie(df['Output (S)'].value_counts(),autopct='%0.2f',labels=df['Output (S)'].value_counts().index)

plt.suptitle('Frequency of both the classes')
plt.show()

plt.bar(x = df.columns , height = df.corr()['Output (S)'])
plt.axhline(y=0, color='black', linewidth=1)
plt.title('Correlation between Ouput(S) and other columns')
plt.show()

ls = ['Ia','Ib','Ic','Va','Vb','Vc']

plt.figure(figsize=(12, 5))
for i in range(2):
    for j in range(3):
        plt.subplot(2, 3, i * 3 + (j + 1))
        sns.kdeplot(df[ls[i * 3 + j]])

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.suptitle('Kde distribution of all columns')
plt.show()

ls = ['Ia','Ib','Ic','Va','Vb','Vc']

plt.figure(figsize=(12, 5))
for i in range(2):
    for j in range(3):
        plt.subplot(2, 3, i * 3 + (j + 1))
        sns.boxplot(x=df[ls[i*3+j]])

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.suptitle('Box plot of all columns')
plt.show()

ls = ['Ia','Ib','Ic','Va','Vb','Vc']

df_melted1 = df[ls[:3]].melt(var_name='Variable', value_name='Value')
df_melted2 = df[ls[3:]].melt(var_name='Variable', value_name='Value')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.stripplot(x='Variable', y='Value', data=df_melted1, ax=ax1)
ax1.set_title('Strip Plot of Ia, Ib, Ic')

sns.stripplot(x='Variable', y='Value', data=df_melted2, ax=ax2)
ax2.set_title('Strip Plot of Va, Vb, Vc')

fig.suptitle('Strip Plots of Different Sets of Variables', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to include the overall title
plt.show()

pair_plot = sns.pairplot(data=df.drop('Output (S)', axis=1))
pair_plot.fig.suptitle('Pair Plot of Features', fontsize=16)
pair_plot.fig.subplots_adjust(top=0.95)

plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.drop('Output (S)',axis=1).corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

ls = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

plt.figure(figsize=(12, 8))

for i in range(2):
    for j in range(3):
        plt.subplot(2, 3, i * 3 + (j + 1))
        stats.probplot(df[ls[i*3+j]], dist="norm", plot=plt)
        plt.title(ls[i*3+j])

plt.tight_layout()
plt.suptitle('QQ plots for all columns', y=1.05, fontsize=16)
plt.show()

x_train,x_test,y_train,y_test = train_test_split(df.drop('Output (S)',axis=1),df['Output (S)'],random_state=42,test_size=0.2)

# Check column types to identify datetime columns
print(x_train.dtypes)
print(x_test.dtypes)

# If datetime columns are present, remove them or convert to numeric
# Remove datetime columns
x_train = x_train.select_dtypes(exclude=['datetime'])
x_test = x_test.select_dtypes(exclude=['datetime'])

# Now, perform scaling on numeric columns
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#Model Training

em = pd.DataFrame(columns=['model_name','accuracy','precision','recall'])

#Logistic Regression

lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
em.loc[em.shape[0]] = ['LogisticRegression',accuracy_score(y_test,y_pred),precision_score(y_test,y_pred),recall_score(y_test,y_pred)]
print(classification_report(y_test,y_pred))

#RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
em.loc[em.shape[0]] = ['RandomForestClassifier',accuracy_score(y_test,y_pred),precision_score(y_test,y_pred),recall_score(y_test,y_pred)]
print(classification_report(y_test,y_pred))


#em

# plt.figure(figsize=(10, 5))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt='d', xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#ANN

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Dropout

model = Sequential([
    Dense(256,'relu',input_dim=x_train.shape[1]),
    Dropout(0.5),
    Dense(128,'relu'),
    Dropout(0.5),
    Dense(1,'sigmoid'),
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_data = (x_test,y_test),epochs=10)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,3))
ax1.plot(history.history['loss'],label='training')
ax1.plot(history.history['val_loss'],label='validation')
ax1.set_title('loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')

ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.set_title('accuracy')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')

plt.subplots_adjust(wspace=0.3)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
plt.show()

#v2 pre-code

#Stability

start_time = datetime.now()

# Data reading
stab_data = pd.read_csv(r"data/smart_grid_stability_augmented.csv")

print(stab_data.head())
print(stab_data.tail())
print(stab_data.isna().sum())




# Features correlation exploring

map1 = {'unstable': 0, 'stable': 1}
stab_data['stabf'] = stab_data['stabf'].replace(map1)
stab_data['stabf'].value_counts(normalize=True)

correlation_matrix = stab_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Map')
plt.show()

stab_data.drop('stab', axis=1, inplace= True)


# splitting  data to train and test

X2 = stab_data.drop('stabf', axis =1)
y2 = stab_data['stabf']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

X2_train = X2_train.values
y2_train = y2_train.values

X2_test = X2_test.values
y2_test = y2_test.values

scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)

# ANN model

model = Sequential([
    Dense(64, activation='relu', input_dim=X2_train.shape[1]),
    Dropout(0.3),  # Dropout for regularization
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  # Binary classification
              metrics=['accuracy'])

history = model.fit(X2_train, y2_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    verbose=1)

# Evaluate on the test set
y2_pred = model.predict(X2_test).round().astype(int)

print("Accuracy on Test Data:", accuracy_score(y2_test, y2_pred))
print("\nClassification Report:\n", classification_report(y2_test, y2_pred))


cm = confusion_matrix(y2_test, y2_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unstable', 'Stable'], yticklabels=['Unstable', 'Stable'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


end_time = datetime.now()
print('\nStart time', start_time)
print('End time', end_time)
print('Time elapsed', end_time - start_time)

# Save the model
model.save("Stability_model.h5")
print("Model saved to 'Stability_model.h5'")

#Fault Detection

start_time = datetime.now()

# Fault detection
'''
Inputs - [Ia,Ib,Ic,Va,Vb,Vc]
Outputs - 0 (No-fault) or 1(Fault is present)
'''

# Data reading
FaultorNot_data = pd.read_csv(r"data/detect_dataset.csv")
FaultorNot_data.dropna(axis=1, inplace=True)
print(FaultorNot_data.head())
print(FaultorNot_data.tail())
print(FaultorNot_data.isna().sum())





# Features correlation exploring
correlation_matrix = FaultorNot_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Map')
plt.show()


# splitting  data to train and test
X2 = FaultorNot_data.drop('Output (S)', axis =1)
y2 = FaultorNot_data['Output (S)']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

X2_train = X2_train.values
y2_train = y2_train.values

X2_test = X2_test.values
y2_test = y2_test.values

scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)

# ANN model

model = Sequential([
    Dense(64, activation='relu', input_dim=X2_train.shape[1]),
    Dropout(0.3),  # Dropout for regularization
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',  # Binary classification
              metrics=['accuracy'])

history = model.fit(X2_train, y2_train,
                    validation_split=0.2,
                    epochs=20,
                    batch_size=32,
                    verbose=1)

# Evaluate on the test set
y2_pred = model.predict(X2_test).round().astype(int)

print("Accuracy on Test Data:", accuracy_score(y2_test, y2_pred))
print("\nClassification Report:\n", classification_report(y2_test, y2_pred))


cm = confusion_matrix(y2_test, y2_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fault', 'No fault'], yticklabels=['Fault', 'No fault'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Save the model
model.save("fault_detection_model.h5")
print("Model saved to 'fault_detection_model.h5'")




# Fault classification
'''
Inputs - [Ia,Ib,Ic,Va,Vb,Vc]
Outputs - [G C B A]
Examples :
[0 0 0 0] - No Fault
[1 0 0 1] - LG fault (Between Phase A and Gnd)
[0 0 1 1] - LL fault (Between Phase A and Phase B)
[1 0 1 1] - LLG Fault (Between Phases A,B and ground)
[0 1 1 1] - LLL Fault(Between all three phases)
[1 1 1 1] - LLLG fault( Three phase symmetrical fault)
'''

# Data reading
Faulttype_data = pd.read_csv(r"data/classData.csv")
print(Faulttype_data.head())
print(Faulttype_data.tail())
print(Faulttype_data.isna().sum())
# Fault mapping
# mapping function
def map_fault(row):
    # No Fault
    if row['G'] == 0 and row['C'] == 0 and row['B'] == 0 and row['A'] == 0:
        return 0

    #LG Fault
    elif row['G'] == 1 and row['C'] == 0 and row['B'] == 0 and row['A'] == 1:
        return 1
    elif row['G'] == 1 and row['C'] == 1 and row['B'] == 0 and row['A'] == 0:
        return 1
    elif row['G'] == 1 and row['C'] == 0 and row['B'] == 1 and row['A'] == 0:
        return 1

    #LL fault
    elif row['G'] == 0 and row['C'] == 0 and row['B'] == 1 and row['A'] == 1:
        return 2
    elif row['G'] == 0 and row['C'] == 1 and row['B'] == 0 and row['A'] == 1:
        return 2
    elif row['G'] == 0 and row['C'] == 1 and row['B'] == 1 and row['A'] == 0:
        return 2

    #LLG Fault
    elif row['G'] == 1 and row['C'] == 0 and row['B'] == 1 and row['A'] == 1:
        return 3
    elif row['G'] == 1 and row['C'] == 1 and row['B'] == 0 and row['A'] == 1:
        return 3
    elif row['G'] == 1 and row['C'] == 1 and row['B'] == 1 and row['A'] == 0:
        return 3

    #LLL fault
    elif row['G'] == 0 and row['C'] == 1 and row['B'] == 1 and row['A'] == 1:
        return 4

    #LLLG Fault
    elif row['G'] == 1 and row['C'] == 1 and row['B'] == 1 and row['A'] == 1:
        return 5



# Apply mapping
Faulttype_data['Fault Type'] = Faulttype_data.apply(map_fault, axis=1)
Faulttype_data.drop(['G', 'C', 'B', 'A'], axis=1, inplace=True)

# Correlation heatmap
correlation_matrix = Faulttype_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Map')
plt.show()

# Splitting data
X2 = Faulttype_data.drop('Fault Type', axis=1)
y2 = Faulttype_data['Fault Type']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

# Standardizing features
scaler = StandardScaler()
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)

# ANN Model for Multiclass Classification
model = Sequential([
    Dense(128, activation='relu', input_dim=X2_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(6, activation='softmax')  # Output layer for 6 classes
])

model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])

# Train the model
history = model.fit(X2_train, y2_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    verbose=1)

# Evaluate on the test set
y2_pred = model.predict(X2_test)
y2_pred_classes = y2_pred.argmax(axis=1)  # Get the class with the highest probability

# Print metrics
print("Accuracy on Test Data:", accuracy_score(y2_test, y2_pred_classes))
print("\nClassification Report:\n", classification_report(y2_test, y2_pred_classes))


# Confusion Matrix
cm = confusion_matrix(y2_test, y2_pred_classes)  # Use y_pred_classes from the previous model
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Fault', 'LG Fault', 'LL Fault', 'LLG Fault', 'LLL Fault', 'LLLG Fault'],
            yticklabels=['No Fault', 'LG Fault', 'LL Fault', 'LLG Fault', 'LLL Fault', 'LLLG Fault'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.show()

# Accuracy and Loss Over Epochs
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Save the model
model.save("fault_classification_model.h5")
print("Model saved to 'fault_classification_model.h5'")

end_time = datetime.now()
print('\nStart time', start_time)
print('End time', end_time)
print('Time elapsed', end_time - start_time)

#Threshold v1

# Define custom threshold for current, overvoltage, and undervoltage
current_threshold = float(input("Enter the current threshold (in k Amperes): "))  # e.g., 2.0 M Amperes
overvoltage_threshold = float(input("Enter the overvoltage threshold multiplier: "))  # e.g., 1.1 for 110%
undervoltage_threshold = float(input("Enter the undervoltage threshold multiplier: "))  # e.g., 0.9 for 90%

# Assume the nominal voltage (e.g., 100kV or 0.1 MV for example purposes)
nominal_voltage = 100  # kV

# Step 2: Check if any of the currents or voltages exceed the threshold
fault_condition_threshold = (
    (df['Ia'].abs() > current_threshold) |
    (df['Ib'].abs() > current_threshold) |
    (df['Ic'].abs() > current_threshold) |
    (df['Va'].abs() > nominal_voltage * overvoltage_threshold) |
    (df['Vb'].abs() > nominal_voltage * overvoltage_threshold) |
    (df['Vc'].abs() > nominal_voltage * overvoltage_threshold) |
    (df['Va'].abs() < nominal_voltage * undervoltage_threshold) |
    (df['Vb'].abs() < nominal_voltage * undervoltage_threshold) |
    (df['Vc'].abs() < nominal_voltage * undervoltage_threshold)
)

# If fault condition is met, label it as a fault (1)
df['Predicted_Fault_Threshold'] = fault_condition_threshold.astype(int)

#faults_df = df[fault_condition_threshold]

# STEP 1: Select Your Best Models (Logistic Regression and Random Forest)
best_model_lr = LogisticRegression()
best_model_rf = RandomForestClassifier()

# Train the Logistic Regression Model
best_model_lr.fit(x_train, y_train)
y_pred_lr = best_model_lr.predict(x_test)

# Train the Random Forest Model
best_model_rf.fit(x_train, y_train)
y_pred_rf = best_model_rf.predict(x_test)

# STEP 2: Create a DataFrame with Predictions from Both Models
# Inspect the shape of x_test to understand the number of columns
print(x_test.shape)

# Adjust the column names accordingly if x_test has 8 columns
# You should replace 'Some_Column_7' and 'Some_Column_8' with the actual column names
column_names = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']  # Adjust as per your dataset
test_df = pd.DataFrame(x_test, columns=column_names)

# Add the corresponding timestamps from the original DataFrame (df)
# Assuming the Timestamp column exists in df and corresponds to x_test's rows
test_df['Timestamp'] = df['Timestamp'].iloc[:len(x_test)].values

# Add the actual labels, predictions from both models
test_df['Actual_Fault'] = y_test.values
test_df['Predicted_Fault_LR'] = y_pred_lr
test_df['Predicted_Fault_RF'] = y_pred_rf

# Create a human-readable status column for both models
test_df['Status_LR'] = test_df['Predicted_Fault_LR'].apply(lambda x: 'Fault' if x == 1 else 'No Fault')
test_df['Status_RF'] = test_df['Predicted_Fault_RF'].apply(lambda x: 'Fault' if x == 1 else 'No Fault')

# View the first few rows to compare the results
print(test_df[['Timestamp', 'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc', 'Actual_Fault', 'Predicted_Fault_LR', 'Status_LR', 'Predicted_Fault_RF', 'Status_RF']].head())


#Using models v2

# Fault Detection

# Load the model
fault_detection_model = load_model("fault_detection_model.h5")
print("Model loaded successfully!")

# Smart Meter data simulation
'''example : Ia = -170.4721962	Ib=9.219613499	Ic=161.2525827	Va=0.054490004	Vb= -0.659920931	Vc=0.605430928,
fault detection:no fault

example : Ia = 56.4184960	Ib=-480.9648594	Ic=427.2942203	Va=0.322317424	Vb= 0.030412678	Vc=-0.352730102
fault detection: fault
'''


Ia = 	-134.4076999
Ib = -688.1012042
Ic = 824.5918837
Va = -0.040103509
Vb = 	0.008221394
Vc = 0.031882115

smart_meter_data = np.array([[Ia, Ib, Ic, Va, Vb, Vc]])

# Apply scaling (if you used StandardScaler during training)
new_data_scaled = scaler.transform(smart_meter_data)  # Use the same scaler as used for training

# Predict using the model
predicted_probabilities = fault_detection_model.predict(new_data_scaled)
predicted_class = np.round(predicted_probabilities)  # Get the predicted class index
predicted_class = predicted_class.item()

# Fault mapping
if predicted_class == 1:
    predicted_fault = "Fault"
elif predicted_class == 0:
    predicted_fault = "No Fault"
print(f"Fault Detection: {predicted_fault}")





# Load the model
fault_classification_model = load_model("fault_classification_model.h5")
print("Model loaded successfully!")


# Fault Classification
if predicted_class == 1:
    # Smart Meter data simulation
    '''example : Ia = -170.4721962	Ib=9.219613499	Ic=161.2525827	Va=0.054490004	Vb= -0.659920931	Vc=0.605430928,
    fault detection:no fault
    '''

    smart_meter_data = np.array([[Ia, Ib, Ic, Va, Vb, Vc]])

    # Apply scaling (if you used StandardScaler during training)
    new_data_scaled = scaler.transform(smart_meter_data)  # Use the same scaler as used for training

    # Predict using the model
    predicted_probabilities = fault_classification_model.predict(new_data_scaled)
    predicted_class = np.argmax(predicted_probabilities, axis=1)  # Get the predicted class index

    # Fault mapping
    fault_types = ['No Fault', 'LG Fault', 'LL Fault', 'LLG Fault', 'LLL Fault', 'LLLG Fault']
    predicted_fault_type = fault_types[predicted_class[0]]

    print(f"Predicted Fault Type: {predicted_fault_type}")
else:
    print("No fault detected")