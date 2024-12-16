# Cat and Dog Image Classifier using CNN

This repository includes all the necessary files and code for my project on image classification using Convolutional Neural Networks (CNN).

---

## 1. Tools and Libraries Needed

To run this project, you will need Python and Jupyter Notebook. Below are the main libraries and tools used:

- **TensorFlow**: To create and train the CNN model.
- **NumPy**: For performing mathematical computations.
- **Matplotlib**: To display and visualize images and results.
- **Pandas**: For handling datasets.
- **os**: To work with files and directories.

Make sure you have these libraries installed before running the project.

---

## 2. Purpose of the Project

This project aims to develop a Convolutional Neural Network that can classify images of cats and dogs. The goal is to build a model that not only works well on the provided training and testing data but also performs accurately on unseen images.

### Questions the Project Answers:
1. How well can CNNs distinguish between images of cats and dogs?
2. Can the trained model correctly identify external images that were not part of the original dataset?

---

## 3. Project Files and Structure

This project consists of the following files and folders:

1. **Main Notebook**: The file `CNN for Image Classification.ipynb` contains all the steps, including:
   - Data preparation and preprocessing.
   - Building, training, and evaluating the CNN model.
   - Making single predictions on unseen images.

2. **Dataset**:  
   The dataset is stored on Google Drive due to its large size.  
   You can access and download it here:  
   [Download Dataset](https://drive.google.com/drive/folders/1pHc80LZqwdBfuvsKEnyEvhtU8oiScFxf?usp=drive_link)  

3. **Read Me.txt**: A detailed description of the project structure and usage.

---

## 4. How to Use the Files

Follow these steps to run and use the project files successfully:

1. **Download the Dataset**:
   - Visit the Google Drive link provided above and download the dataset folder.
   - Ensure the folder structure remains as follows:
     ```
     dataset/
       ├── training_set/
       ├── test_set/
       └── single_prediction/
     ```

2. **Run the Notebook**:
   - Open the `CNN for Image Classification.ipynb` file in Jupyter Notebook or any supported platform (like Google Colab).
   - Execute each cell step by step to:
     - Import necessary libraries.
     - Preprocess the dataset.
     - Train and evaluate the CNN model.
     - Test the model on new images.

3. **Make a Single Prediction**:
   - Update the `image_path` variable in the "Making a Single Prediction" section with the path to any image from the `single_prediction` folder.
   - Example:
     ```python
     image_path = 'dataset/single_prediction/cat_or_dog_1.jpg'
     ```
   - Run the prediction code to see the result (Cat or Dog) along with the displayed image.

---

## 5. Highlights of the Results

The trained CNN achieved excellent performance in identifying whether an image is a cat or a dog. Additionally, the model was tested on images not included in the training or testing datasets, and it correctly predicted the results.

### Key Points:
- The validation accuracy of the model exceeded **80%** during training.
- The model successfully predicted unseen images, including those downloaded from external sources, demonstrating its ability to generalize.

---

## 6. Medium Post

You can read more about the project on Medium, where I explain the entire process, from data preparation to results:  
[Read the Full Project on Medium](<insert your Medium post link here>)

---

## 7. Credits and Information

- **Data**: The dataset for this project was provided as part of my coursework and hosted on Google Drive due to its size.  
- **Author**: This project was created as part of a learning exercise.  
- **Acknowledgments**: Special thanks to my professor for providing the dataset and guidance throughout this project.  


