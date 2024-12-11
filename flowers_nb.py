import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import layers, models
import os
import pickle
import shutil
from sklearn.model_selection import train_test_split

# Set up paths
train_dir = 'train'  # Path to the flowers dataset (train data)

# Define image size and batch size
image_size = (150, 150)
batch_size = 32

# Create a validation directory
val_dir = 'path_to_val_data'
os.makedirs(val_dir, exist_ok=True)

# Function to split train data into train and validation sets
def split_dataset(src_dir, val_dir, val_size=0.2):
    # List all the categories (subdirectories) inside the src_dir
    for category in os.listdir(src_dir):
        category_path = os.path.join(src_dir, category)
        
        # Only process directories (categories)
        if os.path.isdir(category_path):
            # Create category directories in val_dir
            val_category_path = os.path.join(val_dir, category)
            os.makedirs(val_category_path, exist_ok=True)
            
            # Get the list of image files in each category
            images = [img for img in os.listdir(category_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Split the images into train and validation
            train_images, val_images = train_test_split(images, test_size=val_size, random_state=42)
            
            # Move the images into their respective directories
            for image in train_images:
                shutil.move(os.path.join(category_path, image), os.path.join(src_dir, category, image))
            for image in val_images:
                shutil.move(os.path.join(category_path, image), os.path.join(val_category_path, image))

# Split the dataset (uncomment to execute splitting)
split_dataset(train_dir, val_dir, val_size=0.2)

# Data Preprocessing with ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize image values to [0,1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # Assuming 5 flower classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    verbose=1
)

# Save the model in Keras .h5 format
model.save('flower_classification_model.h5')

# Save the model in Pickle format
with open('flower_classification_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the Keras model
loaded_model = tf.keras.models.load_model('flower_classification_model.h5')

# Load the Pickle model
with open('flower_classification_model.pkl', 'rb') as f:
    loaded_model_pkl = pickle.load(f)

# Example function for predicting an image
def predict_image(model, img_path):
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0  # Normalize image
    img_array = img_array.reshape((1, 150,150, 3))  # Add batch dimension
    prediction = model.predict(img_array)
    return prediction

# Example prediction with loaded models
# result_keras = predict_image(loaded_model, 'path_to_image.jpg')
# result_pkl = predict_image(loaded_model_pkl, 'path_to_image.jpg')

