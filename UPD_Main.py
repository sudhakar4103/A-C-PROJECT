# ----------------- IMPORT PACKAGES ------------------------

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter.filedialog import askopenfilename

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import matplotlib.image as mpimg



# ------------------------- READ INPUT IMAGE -------------------------


filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)

plt.axis ('off')
# plt.savefig("Ori.png")
plt.title('Original Image')
plt.show()


# ------------------------- PREPROCESS -------------------------

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   
         
#==== GRAYSCALE IMAGE ====



SPV = np.shape(img)

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1)
plt.axis ('off')
plt.show()




# ------------------------- 3.FEATURE EXTRACTION -------------------------


#=== MEAN STD DEVIATION ===

mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
features_extraction = [mean_val,median_val,var_val]

print("-------------------------------------")
print("        Feature Extraction          ")
print("-------------------------------------")
print()
print(features_extraction)



# === GRAY LEVEL CO OCCURENCE MATRIX ===

from skimage.feature import graycomatrix, graycoprops

print()
print("-----------------------------------------------------")
print("FEATURE EXTRACTION -->GRAY LEVEL CO-OCCURENCE MATRIX ")
print("-----------------------------------------------------")
print()


PATCH_SIZE = 21

# open the image

image = img[:,:,0]
image = cv2.resize(image,(768,1024))
 
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
    xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(graycoprops(glcm, 'correlation')[0, 0])


# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('GLCM')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')
plt.show()

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Region 1')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Region 2')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()
plt.show()


sky_patches0 = np.mean(sky_patches[0])
sky_patches1 = np.mean(sky_patches[1])
sky_patches2 = np.mean(sky_patches[2])
sky_patches3 = np.mean(sky_patches[3])

Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
Tesfea1 = []
Tesfea1.append(Glcm_fea[0])
Tesfea1.append(Glcm_fea[1])
Tesfea1.append(Glcm_fea[2])
Tesfea1.append(Glcm_fea[3])


print("---------------------------------------------------")
print("GLCM FEATURES =")
print("---------------------------------------------------")
print()
print(Glcm_fea)


# ------------------------- 4. IMAGE SPLITTING -------------------------
    
#==== TRAIN DATA FEATURES ====

import pickle

with open('dot.pickle', 'rb') as f:
    dot1 = pickle.load(f)
  

import pickle
with open('labels.pickle', 'rb') as f:
    labels1 = pickle.load(f) 


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print("---------------------------------")
print("Image Splitting")
print("---------------------------------")
print()
print("1. Total Number of images =", len(dot1))
print()
print("2. Total Number of Test  =", len(x_test))
print()
print("3. Total Number of Train =", len(x_train))    



# ------------------------- CLASSIFICATION -------------------------

# --- DIMENSION EXPANSION


from keras.utils import to_categorical
   
y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1, num_classes=6)
test_Y_one_hot = to_categorical(y_test1, num_classes=6)

# FIX 1: Correctly copy x_train data to x_train2
x_train2 = np.zeros((len(x_train), 50, 50, 3))
for i in range(len(x_train)):
    # Check if the image has 3 channels, if not, convert grayscale to 3-channel
    if len(x_train[i].shape) == 2:
        # Convert grayscale to 3-channel by repeating
        x_train2[i] = np.stack([x_train[i]] * 3, axis=-1)
    elif x_train[i].shape[-1] == 1:
        # Convert single channel to 3-channel
        x_train2[i] = np.repeat(x_train[i], 3, axis=-1)
    else:
        # Already 3-channel, just copy
        x_train2[i] = x_train[i]

# Also prepare test data properly
x_test2 = np.zeros((len(x_test), 50, 50, 3))
for i in range(len(x_test)):
    if len(x_test[i].shape) == 2:
        x_test2[i] = np.stack([x_test[i]] * 3, axis=-1)
    elif x_test[i].shape[-1] == 1:
        x_test2[i] = np.repeat(x_test[i], 3, axis=-1)
    else:
        x_test2[i] = x_test[i]

# Normalize the images
x_train2 = x_train2.astype('float32') / 255.0
x_test2 = x_test2.astype('float32') / 255.0

# Prepare input image for prediction
input_image_for_prediction = np.zeros((1, 50, 50, 3))
if len(img_resize_orig.shape) == 2:
    input_image_for_prediction[0] = np.stack([img_resize_orig] * 3, axis=-1)
elif img_resize_orig.shape[-1] == 1:
    input_image_for_prediction[0] = np.repeat(img_resize_orig, 3, axis=-1)
else:
    input_image_for_prediction[0] = img_resize_orig
input_image_for_prediction = input_image_for_prediction.astype('float32') / 255.0



# ----------------------------------------------------------------------
# EfficientNet
# ----------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models

# Define input shape
input_shape = (50, 50, 3)

# Load EfficientNetB0 without the top layer
efficient_net = tf.keras.applications.EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    input_shape=input_shape
)

# Freeze the layers of EfficientNet
for layer in efficient_net.layers:
    layer.trainable = False

# Define input layer
input_layer = layers.Input(shape=input_shape)

# Pass input through EfficientNet
efficient_output = efficient_net(input_layer)

# Global Average Pooling
flattened_output = layers.GlobalAveragePooling2D()(efficient_output)

# Fully connected layers
dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(6, activation='softmax')(dense_layer)

# Build model
model_eff = models.Model(inputs=input_layer, outputs=output_layer)

# Compile model
model_eff.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model_eff.summary()

print("-------------------------------------")
print(" EfficientNetB0")
print("-------------------------------------")
print()

# Fit the model
history_eff = model_eff.fit(x_train2, train_Y_one_hot, batch_size=16, epochs=10, verbose=1, validation_data=(x_test2, test_Y_one_hot))

# Evaluate the model
test_loss_eff, test_accuracy_eff = model_eff.evaluate(x_test2, test_Y_one_hot, verbose=1)

loss_eff = history_eff.history['loss']
val_loss_eff = history_eff.history['val_loss']

error_efficient = min(loss_eff)
acc_efficient = test_accuracy_eff * 100

# For demonstration, using actual predictions to calculate metrics
from sklearn.metrics import precision_score, recall_score, f1_score

# Get predictions
y_pred_eff = model_eff.predict(x_test2)
y_pred_classes_eff = np.argmax(y_pred_eff, axis=1)
y_true_classes = np.argmax(test_Y_one_hot, axis=1)

# Calculate metrics
precision_eff = precision_score(y_true_classes, y_pred_classes_eff, average='weighted')
recall_eff = recall_score(y_true_classes, y_pred_classes_eff, average='weighted')
f1_score_eff = f1_score(y_true_classes, y_pred_classes_eff, average='weighted')

print("-------------------------------------")
print("EFFICIENTNET PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Test Accuracy   =", acc_efficient,'%')
print()
print("2. Error Rate =", error_efficient)
print()
print("3. Precision   =", precision_eff*100,'%')
print()
print("4. Recall      =", recall_eff*100,'%')
print()
print("5. F1-score    =", f1_score_eff*100)




# ----------------------------------------------------------------------
# MobileNetV3
# ----------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models

# Define input shape
input_shape = (50, 50, 3)

# Load MobileNetV3Small without the top layer
mobilenet = tf.keras.applications.MobileNetV3Small(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
)

# Freeze all layers
for layer in mobilenet.layers:
    layer.trainable = False

# Define input layer
input_layer = layers.Input(shape=input_shape)

# Pass input through MobileNetV3
mobilenet_output = mobilenet(input_layer)

# Global Average Pooling
flattened_output = layers.GlobalAveragePooling2D()(mobilenet_output)

# Fully connected layers
dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(6, activation='softmax')(dense_layer)

# Build the model
model_mobilenet = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model_mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model_mobilenet.summary()

print("-------------------------------------")
print(" MobileNetV3Small")
print("-------------------------------------")
print()

# Fit the model
history_mobilenet = model_mobilenet.fit(x_train2, train_Y_one_hot, batch_size=16, epochs=10, verbose=1, validation_data=(x_test2, test_Y_one_hot))

# Evaluate the model
test_loss_mobilenet, test_accuracy_mobilenet = model_mobilenet.evaluate(x_test2, test_Y_one_hot, verbose=1)

loss_mobilenet = history_mobilenet.history['loss']
val_loss_mobilenet = history_mobilenet.history['val_loss']

error_mobilenet = min(loss_mobilenet)
acc_mobilenet = test_accuracy_mobilenet * 100

# Get predictions for MobileNet
y_pred_mobilenet = model_mobilenet.predict(x_test2)
y_pred_classes_mobilenet = np.argmax(y_pred_mobilenet, axis=1)

# Calculate metrics
precision_mn = precision_score(y_true_classes, y_pred_classes_mobilenet, average='weighted')
recall_mn = recall_score(y_true_classes, y_pred_classes_mobilenet, average='weighted')
f1_score_mn = f1_score(y_true_classes, y_pred_classes_mobilenet, average='weighted')

print("-------------------------------------")
print("MOBILENET PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Test Accuracy   =", acc_mobilenet,'%')
print()
print("2. Error Rate =", error_mobilenet)
print()
print("3. Precision   =", precision_mn*100,'%')
print()
print("4. Recall      =", recall_mn*100,'%')
print()
print("5. F1-score    =", f1_score_mn*100)




# ------------------------ METRICS COMPARISON TABLE ------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Create a dictionary with metrics
metrics_dict = {
    "Algorithm": ["EfficientNetB0", "MobileNetV3Small"],
    "Accuracy (%)": [acc_efficient, acc_mobilenet],
    "Precision (%)": [precision_eff*100, precision_mn*100],
    "Recall (%)": [recall_eff*100, recall_mn*100],
    "F1-score (%)": [f1_score_eff*100, f1_score_mn*100]
}

# Create a DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Display table
print("\n-------------------------------------")
print("COMPARISON TABLE")
print("-------------------------------------")
print(metrics_df)

# ------------------------ ACCURACY COMPARISON GRAPH ------------------------

plt.figure(figsize=(8,5))
plt.bar(metrics_df['Algorithm'], metrics_df['Accuracy (%)'], color=['skyblue', 'lightgreen'])
plt.title('Accuracy Comparison Between Algorithms')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, v in enumerate(metrics_df['Accuracy (%)']):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
plt.show()

# ------------------------ TRAINING LOSS COMPARISON ------------------------

plt.figure(figsize=(10,5))
plt.plot(history_eff.history['loss'], label='EfficientNet Training Loss', color='skyblue')
plt.plot(history_eff.history['val_loss'], label='EfficientNet Validation Loss', color='skyblue', linestyle='--')
plt.plot(history_mobilenet.history['loss'], label='MobileNet Training Loss', color='lightgreen')
plt.plot(history_mobilenet.history['val_loss'], label='MobileNet Validation Loss', color='lightgreen', linestyle='--')
plt.title('Training and Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



# -----------------------------------------------------------------
# FIX 2: Use actual model prediction instead of mean comparison
# -----------------------------------------------------------------

class_names = {
    0: 'Normal',
    1: 'Cataract',   
    2: 'Anemia',
    3: 'NoAnemia',
}

# Use the trained model (EfficientNet) to predict the input image
print("\n----------------------------------------")
print("PREDICTING INPUT IMAGE USING TRAINED MODEL")
print("----------------------------------------")

# Make prediction using EfficientNet model
predictions = model_eff.predict(input_image_for_prediction)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

if predicted_class in class_names:
    print(f"Identified as: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    identified_class = predicted_class
else:
    print("Class not recognized.")
    identified_class = -1

print("----------------------------------------")

# Also show prediction probabilities for all classes
print("\nClass Probabilities:")
for class_id, class_name in class_names.items():
    probability = predictions[0][class_id] * 100
    print(f"  {class_name}: {probability:.2f}%")

# Display the input image with prediction
plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
plt.axis('off')
plt.title(f'Prediction: {class_names[predicted_class]} ({confidence:.1f}% confidence)')
plt.show()

# Compare with MobileNet prediction
print("\n----------------------------------------")
print("MOBILENET PREDICTION FOR COMPARISON")
print("----------------------------------------")

mobilenet_predictions = model_mobilenet.predict(input_image_for_prediction)
mobilenet_predicted_class = np.argmax(mobilenet_predictions[0])
mobilenet_confidence = np.max(mobilenet_predictions[0]) * 100

if mobilenet_predicted_class in class_names:
    print(f"MobileNet identifies as: {class_names[mobilenet_predicted_class]}")
    print(f"Confidence: {mobilenet_confidence:.2f}%")
    
    if predicted_class == mobilenet_predicted_class:
        print("✓ Both models agree on the prediction")
    else:
        print("✗ Models disagree on the prediction")
        print(f"  EfficientNet: {class_names[predicted_class]}")
        print(f"  MobileNet: {class_names[mobilenet_predicted_class]}")
print("----------------------------------------")