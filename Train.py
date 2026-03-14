# ====================== IMPORT PACKAGES ======================
import os
import numpy as np
import cv2
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import pickle



# DATASET_PATH = "Dataset"  

# # Get list of subdirectories (i.e., class names)
# classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]


# # ====================== DATA LOADING FUNCTION ======================

# def load_images_from_folders(base_path, class_list, image_size=(50, 50)):
#     data = []
#     labels = []
    
#     for idx, class_name in enumerate(class_list):
#         class_path = os.path.join(base_path, class_name)
#         if not os.path.isdir(class_path):
#             print(f"Skipping {class_path}, not a folder.")
#             continue

#         for img_file in os.listdir(class_path):
#             img_path = os.path.join(class_path, img_file)
#             try:
#                 img = mpimg.imread(img_path)
#                 img = cv2.resize(img, image_size)

#                 # Convert to grayscale
#                 try:
#                     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 except:
#                     gray = img  # Already grayscale or other format

#                 data.append(np.array(gray))
#                 labels.append(idx)
#             except Exception as e:
#                 print(f"Error loading image {img_path}: {e}")
#                 continue
#     return np.array(data), np.array(labels)



d1 = os.listdir('Dataset/1_normal')
  
d2 = os.listdir('Dataset/2_cataract')
  
d3 = os.listdir('Dataset/Anemia')

#2
  
d4 = os.listdir('Dataset/NoAnemia')



dot1= []
labels1 = []
  
for img in d1:
      # print(img)
    try:
      img_1 = mpimg.imread('Dataset/1_normal/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(0)
    except:
        None
      
for img in d2:
      # print(img)
    try:
      img_1 = mpimg.imread('Dataset/2_cataract/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(1)      
    except:
        None      
      
for img in d3:
      # print(img)
    try:
      img_1 = mpimg.imread('Dataset/Anemia/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(2)            
      
    except:
        None      
      
 # ----- 2

   
for img in d4:
      # print(img)
    try:
      img_1 = mpimg.imread('Dataset/NoAnemia/' + "/" + img)
      img_1 = cv2.resize(img_1,((50, 50)))
  
  
  
      try:            
          gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
          
      except:
          gray = img_1
  
      
      dot1.append(np.array(gray))
      labels1.append(3)
    except:
        None     








# ====================== LOAD AND SAVE DATA ======================

# X, y = load_images_from_folders(DATASET_PATH, classes)
# 
# Save data and labels using pickle
with open('dot.pickle', 'wb') as f:
    pickle.dump(dot1, f)

with open('labels.pickle', 'wb') as f:
    pickle.dump(labels1, f)

print("Data and labels saved successfully!")
