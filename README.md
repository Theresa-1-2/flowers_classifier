# Flower Classification using CNN

This project classifies images of flowers into one of five categories using a Convolutional Neural Network (CNN). The dataset consists of images stored in subdirectories where each subdirectory represents a different flower class. The images are split into training and validation sets, with 80% for training and 20% for validation.

### Data Preprocessing
The dataset is preprocessed using TensorFlow's `ImageDataGenerator` to normalize pixel values and apply data augmentation (shear, zoom, horizontal flip) to the training data to avoid overfitting.

### Model Architecture
A CNN model is built with:
- Convolutional layers for feature extraction.
- Max-pooling layers for downsampling.
- Dense layers for classification.
- Dropout layer to prevent overfitting.

The model is compiled with the Adam optimizer and categorical cross-entropy loss function. It is trained for 10 epochs.

### Saving and Loading the Model
After training, the model is saved in both `.h5` (Keras) and `.pkl` (Pickle) formats for easy reuse.

### Prediction
The trained model can classify new flower images using the `predict_image()` function.

### Requirements
- TensorFlow
- Keras
- scikit-learn
- Python 3.x

### License
This project is licensed under the MIT License.

