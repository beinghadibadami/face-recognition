# **🔍 Face Recognition using CNN & OpenCV**  

## **📌 Overview**  
This project is a **CNN-based Facial Recognition System** that identifies individuals based on facial features. The model is trained **from scratch** using **famous personalities**, **myself**, and **my friend**. The dataset was created by collecting images from the internet and personal images.  

## **🛠️ Features**  
✅ **Trained from Scratch** – Uses CNN (Convolutional Neural Networks) for face recognition.  
✅ **OpenCV for Face Detection** – Automatically detects and crops faces from images.  
✅ **Data Augmentation** – Generates additional images to improve model accuracy.  
✅ **Custom Dataset** – Includes a limited number of persons but can be fine-tuned with new images.  
✅ **High Accuracy** – Achieves **~93% accuracy** on test data.  
✅ **Extendable** – Can be modified for various real-world applications.  

---

## **🔧 Tech Stack**  
🔹 **Python** – The core programming language  
🔹 **TensorFlow & Keras** – For building and training the CNN model  
🔹 **OpenCV** – For face detection and preprocessing  
🔹 **Matplotlib & Seaborn** – For visualization and model evaluation  
🔹 **NumPy & Pandas** – For handling image data  

---

## **📂 Dataset Preparation**  
📌 The dataset consists of **famous personalities** and my own images.  
📌 Images are processed using **OpenCV’s Haarcascade classifier** to crop faces.  
📌 **Data Augmentation** was applied to increase dataset size to 100 images per person.  

### **✨ Steps Taken:**  
1. **Collected images** from the internet & personal images 📸  
2. **Cropped faces** using OpenCV’s `detectMultiScale()`  
3. **Applied Data Augmentation** (rotation, flipping, brightness adjustment)  
4. **Resized images** to `128x128` for CNN input  
5. **Normalized images** (pixel values scaled between 0-1)  
6. **Encoded labels** for different persons  

---

## **🧠 Model Architecture (CNN)**
The CNN model consists of **convolutional layers**, **max-pooling layers**, and **fully connected layers**.  

```
Input (128x128x3) ➡️ Conv2D(32 filters) ➡️ MaxPooling  
➡️ Conv2D(64 filters) ➡️ MaxPooling  
➡️ Conv2D(128 filters) ➡️ MaxPooling  
➡️ Flatten ➡️ Dense(128) ➡️ Dropout(0.5) ➡️ Dense(num_classes, softmax)
```

📌 **Activation Function**: ReLU for hidden layers, Softmax for output  
📌 **Loss Function**: Categorical Crossentropy  
📌 **Optimizer**: Adam  

---

## **📊 Model Performance**  
✔️ **Accuracy**: ~93%  
✔️ **Precision, Recall & F1-score**: Evaluated using a **classification report**  
✔️ **Confusion Matrix**: Plotted for class-wise performance analysis  

---

## **🛠️ How to Use the Model?**  
### **1️⃣ Train the Model**  
Run the following script to train the model from scratch:  
```bash
python train_model.py
```

### **2️⃣ Test with New Images**  
You can test the model by running:  
```python
python predict.py --image_path "test_image.jpg"
```

### **3️⃣ Save & Load Model**  
To **save** the trained model:  
```python
model.save("face_recognition_model.h5")
```
To **load** the saved model later:  
```python
from tensorflow.keras.models import load_model
model = load_model("face_recognition_model.h5")
```

---

## **📂 How to Fine-Tune with Your Own Images?**  
🔹 **Step 1:** Collect and store new images in a folder (e.g., `"new_faces"`)  
🔹 **Step 2:** Merge new images with the existing dataset  
🔹 **Step 3:** Retrain the model using both old and new images  
🔹 **Step 4:** Save the updated model  

Run:  
```python
python train_model.py --dataset "dataset_aug"
```

---

## **📌 Future Improvements & Use Cases**  
### 🚀 **Possible Enhancements:**  
✅ **Live Face Recognition using Webcam** 🎥  
✅ **Real-Time Attendance System** 🏫  
✅ **Secure Login System** 🔐  
✅ **AI-Based Access Control for Offices** 🏢  
✅ **Automated Customer Identification in Shops** 🛒  

---

## **📜 Requirements**  
Install the required dependencies using:  
```bash
pip install -r requirements.txt
```
**`requirements.txt` includes:**  
```
tensorflow
keras
numpy
opencv-python
matplotlib
seaborn
pandas
```

---

## **🔗 Conclusion**  
This project demonstrates how **deep learning & OpenCV** can be used for face recognition. The model is **scalable**, allowing users to **fine-tune it** with their own images. Future updates may include **real-time webcam detection** and **face verification systems**.  

💡 **Feel free to contribute & improve this project!** 😊  
