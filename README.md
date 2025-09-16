# 🩺 Pneumonia Detection from Chest X-Rays

This project uses deep learning to classify chest X-ray images as either showing signs of pneumonia or being normal. It serves as a practical application of Convolutional Neural Networks (CNNs) for a real-world medical imaging problem. Two approaches are explored: a custom CNN built from scratch and a more powerful model using Transfer Learning with VGG16.



---

## Project Goal

The primary objective is to build a reliable model that can accurately distinguish between "Pneumonia" and "Normal" chest X-rays. The project emphasizes the importance of evaluation metrics beyond simple accuracy, such as **Precision** and **Recall**, which are critical in medical diagnostic tasks where a false negative can have serious consequences.

---

## Dataset

The project utilizes the **"Chest X-Ray Images (Pneumonia)"** dataset available on Kaggle.

* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **Description:** The dataset contains 5,863 JPEG images organized into training, validation, and testing sets.
* **Classes:** Pneumonia, Normal.

---

## Key Concepts & Technologies

This project applies several key concepts in deep learning and computer vision:

* **Binary Image Classification:** A core computer vision task.
* **Custom CNN Architecture:** Building a network from scratch to understand the fundamentals of `Conv2D`, `MaxPooling2D`, and `Dense` layers.
* **Transfer Learning:** Leveraging a pre-trained model (VGG16) to improve performance and reduce training time.
* **Data Augmentation:** Artificially expanding the training dataset using `ImageDataGenerator` to prevent overfitting.
* **Handling Class Imbalance:** Using `class_weight` to address the unequal distribution of classes in the dataset.
* **Advanced Evaluation Metrics:** Analyzing model performance using a Confusion Matrix, Precision, Recall, and F1-Score.
* **Visualization:** Using Matplotlib for plotting training history and TensorBoard for real-time monitoring of metrics.

**Technologies Used:**
* Python
* TensorFlow & Keras
* Scikit-learn
* NumPy
* Matplotlib & Seaborn

---

## Workflow

1.  **Data Exploration & Preprocessing:** The images were loaded, visualized, and resized to a uniform size (150x150 pixels). Pixel values were normalized.
2.  **Data Augmentation:** The training data was augmented with random rotations, shifts, shears, and zooms.
3.  **Model 1: Custom CNN:** A sequential CNN was built and trained to establish a baseline performance.
4.  **Model 2: Transfer Learning:** The VGG16 model, pre-trained on ImageNet, was used as a convolutional base. A new classifier head was added and trained on the X-ray data.
5.  **Evaluation:** Both models were evaluated on the test set, with a focus on recall for the "Pneumonia" class.

---

## Model Results

A comparison of the two models on the test set highlights the effectiveness of transfer learning.

### Model 1: Custom CNN

This model served as a solid baseline, achieving an overall accuracy of 90%. However, its recall for "Normal" cases was lower, indicating a tendency to misclassify healthy patients as having pneumonia.

**Classification Report:**
```
              precision    recall  f1-score   support

      NORMAL       0.93      0.78      0.85       234
   PNEUMONIA       0.88      0.96      0.92       390

    accuracy                           0.90       624
   macro avg       0.90      0.87      0.88       624
weighted avg       0.90      0.90      0.89       624
```

### Model 2: Transfer Learning (VGG16)

The Transfer Learning model demonstrated superior and more balanced performance.

**Final Test Set Performance:**
* **Accuracy:** 91.7%
* **Precision:** 90.2%
* **Recall (for Pneumonia):** 97.2%

The standout result is the **97.2% recall**, which means the model successfully identified over 97% of all actual pneumonia cases. This is a crucial outcome for a medical screening tool, as it minimizes the risk of false negatives.

---

## Key Learnings

* **Effectiveness of Transfer Learning:** The pre-trained VGG16 model significantly outperformed the custom CNN. This demonstrates the power of leveraging knowledge from large, general-purpose datasets (like ImageNet) for specialized tasks with smaller datasets.
* **Importance of Recall in Medical AI:** For a diagnostic task, maximizing recall for the positive class (Pneumonia) is more critical than overall accuracy. A model that safely identifies nearly all sick individuals, even at the cost of a few false alarms, is clinically more valuable.
* **Handling Data Imbalance:** Using techniques like `class_weight` was essential. Without it, the model would be biased towards the majority class ("Pneumonia"), leading to poor performance on the minority class ("Normal").
* **Data Augmentation as a Necessity:** Augmentation was vital for both models to improve generalization and prevent overfitting on the limited training data.

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the dataset** from the Kaggle link above and place it in a `data/` directory.
5.  **Run the Jupyter Notebook** to see the full process of data preparation, model training, and evaluation.
