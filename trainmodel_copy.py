# from urllib import request
import json
import sys
from flask import Flask, request, render_template,redirect, jsonify

import matplotlib.pyplot as plt
import base64
import io
import tensorflow as tf
import pathlib
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit' 
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image
import numpy as np
import tensorflow.lite as tflite
import matplotlib
matplotlib.use('agg')  # Set Matplotlib to use the 'agg' backend
from flask import Flask, jsonify, request, render_template

import pathlib

from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dropout  # Add this import
from tensorflow.keras.layers import Dense , BatchNormalization
import signal
import subprocess
# from waitress import serve
from sklearn.utils import resample

tf.config.optimizer.set_jit(True)
# from imblearn.over_sampling import SMOTE
# from imblearn.keras import BalancedBatchGenerator

# Create a Flask web application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_upload'
# app.config['STATIC_FOLDER'] = 'static'


def encode_file_to_base64(file_path):
    with open(file_path, "rb") as file:
        encoded_data = base64.b64encode(file.read()).decode('utf-8')
    return encoded_data

def create_and_save_plot(title, ylabel, xlabel, historyAcc, historyValAcc, epochs):
    # Plot training and validation accuracy
    plt.figure()
    plt.plot(historyAcc)
    plt.plot(historyValAcc)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.xticks(range(0, epochs + 5, 5))

    # Save the accuracy plot as an image
    accuracy_img = io.BytesIO()
    plt.savefig(accuracy_img, format='png')
    accuracy_img.seek(0)
    accuracy_img_base64 = base64.b64encode(accuracy_img.read()).decode('utf-8')

    return accuracy_img_base64


def delete_datasets_folder(data_dir):
    try:
        
        # Iterate over all items in the datasets folder
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)

            for folder in dirs:
                folder_path = os.path.join(root, folder)
                shutil.rmtree(folder_path)

        # Remove the datasets folder itself
        shutil.rmtree(data_dir)

        return True, "Datasets folder and its contents deleted successfully."

    except Exception as e:
        return False, f"Error deleting datasets folder: {str(e)}"

def resize_image(image_path, max_size=300):
    try:
        image = Image.open(image_path)
          # Use LANCZOS filter for resizing
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        image.save(image_path)
    except Exception as e:
        print(f"Error resizing image {image_path}: {str(e)}")

def download_images_from_api(api_url, api_key, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = {
        'api_key': api_key  # Replace 'api_key' with the actual parameter name expected by the API
    }

    response = requests.post(api_url, data=data)

    if response.status_code == 200:
        
        data = json.loads(response.text)

        for label, image_urls in data.items():
            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            for image_url in image_urls:
                try:
                    image_filename = os.path.join(label_dir, os.path.basename(image_url))
                    if not os.path.exists(image_filename):
                        image_response = requests.get(image_url)
                        if image_response.status_code == 200:
                            with open(image_filename, 'wb') as image_file:
                                image_file.write(image_response.content)
                            
                             # Resize the image
                            resize_image(image_filename, max_size=300)
                        else:
                            print(f"Failed to download image: {image_url}")
                    else:
                        print(f"Skipping existing image: {image_filename}")
                except requests.exceptions.MissingSchema as e:
                    # print(f"Error: Invalid image URL - {image_url}")
                    return 'Error: Invalid image URL'
        
    else:
        if response.status_code == 401:
            print ('Error: Invalid API key')

            return 'Error: Invalid API key'
        else:
            print (f'Error: Request failed with status code {response.status_code}')
            return f'Error: Request failed with status code {response.status_code}'
    
    return 'Success: Images downloaded successfully'

# EpochStatusCallback
class EpochStatusCallback(tf.keras.callbacks.Callback):
    def __init__(self,total_steps , current_step,epochs):
        self.total_steps = total_steps
        self.current_step = current_step
        self.epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.current_step += 1
    
        overall_progress_percentage = int((self.current_step / self.total_steps) * 100)
        with open('status.txt', 'w') as f:  # Use 'a' (append) mode instead of 'w' (write)
            f.write(f'Extracting features {epoch + 1}/{self.epochs}  ({overall_progress_percentage}%)')


@app.route('/flask/')
def index():
    with open('running_status.txt', 'r') as f:
        content = f.read()

        if "Stopping" in content:
            with open('running_status.txt', 'w') as f:
                f.write(str(f'Not Running'))
            with open('status.txt', 'w') as f:
                f.write(str(f'Initializing 0%'))
    return render_template('train.html')


@app.route('/flask/status')
def status():
    with open('status.txt', 'r') as f:
        content = f.read()
    return content

@app.route('/flask/running_status')
def running_status():
    with open('running_status.txt', 'r') as f:
        content = f.read()
    return content

@app.route('/flask/restart', methods=['POST'])
def restart():
    
        # if "Not Running" in content:
        #     with open('running_status.txt', 'w') as f:
        #                 f.write(str(f'Not Running'))
        #     previous_url = request.referrer
        #     return redirect(previous_url)
        # else:
    # subprocess.Popen(['waitress-serve --listen=0.0.0.0:5000 trainmodel_copy:app'])
    
    # subprocess.Popen(['python', 'trainmodel_copy.py'])
    # time.sleep(0.5)

    # subprocess.Popen(['waitress-serve', '--listen=0.0.0.0:5000', 'trainmodel_copy:app'])
    subprocess.Popen(['python', 'trainmodel_copy.py'])

    with open('running_status.txt', 'w') as f:
                f.write(str(f'Stopping'))
    with open('status.txt', 'w') as f:
                f.write(str(f'Stopping'))
        
    
    
    os.kill(os.getpid(), signal.SIGTERM)
    os.execv(sys.executable, [sys.executable] + sys.argv)
    return ''






@app.route('/flask/upload', methods=[ 'POST'])
def upload():
  if request.method == 'POST':
        # API endpointhttps://pechai-com.preview-domain.com/img-api.php
        api_url = 'http://pechai.site/model-api.php'

        # File paths
        labels_file_path = 'flask_model/labels.txt'
        model_file_path = 'flask_model/model.tflite'

        # Encode files to base64
        labels_base64 = encode_file_to_base64(labels_file_path)
        model_base64 = encode_file_to_base64(model_file_path)

        # Prepare data for the POST request
        data = {
            'labels': labels_base64,
            'model': model_base64
        }

        # Send POST request to the API endpoint with files
        response = requests.post(api_url, data=data)

        # Print the response from the server
        return response.text

    

@app.route('/flask/test-result', methods=['POST'])
def test_model():

    # Define the image path
    # Check if the post request has the file part
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file temporarily
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(upload_path)

    try:
        # Image processing
        image_size = 224  # Assuming the model expects input size of 224x224
        image = Image.open(upload_path).resize((image_size, image_size))
        input_image = np.array(image, dtype=np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        # Load the TFLite model
        interpreter = tflite.Interpreter(model_path='flask_model/model.tflite')
        interpreter.allocate_tensors()

        # Set input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_image)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])

        # Load the labels from the model
        label_file = 'flask_model/labels.txt'
        with open(label_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        prediction = output.flatten()
        max_index = np.argmax(prediction)
        predicted_disease = labels[max_index]
        confidence = prediction[max_index] * 100.0

        # Print the results
        # result_message = "Predicted Disease: {}\nConfidence: {:.2f}%".format(predicted_disease, confidence)
        result_data = {
            "predicted_disease": predicted_disease,
            "confidence": confidence
        }

    finally:
        # Delete the temporary uploaded image
        os.remove(upload_path)
    return jsonify(result=result_data)
    
    # return result_message
    # return render_template('test_result.html',result=result_message)



@app.route('/flask/train_model', methods=['POST'])
def train_model():
    with open('running_status.txt', 'w') as f:
                f.write(str(f'Running'))
    epochs = 10
    total_steps = 10+epochs # Define the total number of steps in your process

    current_step = 0  #
    ###############################################################
    current_step += 1
    progress_percentage = int((current_step / total_steps) * 100)
    with open('status.txt', 'w') as f:
                f.write(str(f'Initializing {progress_percentage}%'))
    ###############################################################
    data_dir = 'datasets'  # Replace with your dataset directory
    success, message = delete_datasets_folder(data_dir)
    # if success:
    #     print(message)
    # else:
    #     print(f"Error: {message}")

    #SHOW : deleted in html
    # Call the function to download images from the API
    api_url = 'http://pechai.site/img-api.php'
    api_key = request.form.get('apiKey')
    ###############################################################
    current_step += 1
    progress_percentage = int((current_step / total_steps) * 100)
    with open('status.txt', 'w') as f:
                f.write(str(f'Fetching datasets {progress_percentage}%'))
    ###############################################################
    result = ""
    try:
        result = download_images_from_api(api_url, api_key, data_dir)
    # Continue with the rest of your code if the function call is successful
    except Exception as e: 
        result = f'Error: {e}'
        with open('status.txt', 'w') as f:
                f.write(str(result))
        with open('running_status.txt', 'w') as f:
                f.write(str(result))  
        # result = download_images_from_api(api_url, api_key, data_dir)
    #SHOW : download in html
    
    if result.startswith('Error'):
        with open('status.txt', 'w') as f:
                f.write(str(result))
        with open('running_status.txt', 'w') as f:
                f.write(str(result))        
        result_data = {
            "success": 'false', 
            "message": result,
        }
        return jsonify(result=result_data)
    else:
        data_dir = pathlib.Path(data_dir)

        # Now, you can access files and folders inside the datasets folder
        all_files = list(data_dir.glob('*/*.jpg'))

        # Rest of your code remains the same
        ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Splitting datasets  {progress_percentage}%'))
        ###############################################################
        # Split the dataset into train, validation, and test sets
        train_files, test_val_files = train_test_split(
            all_files, test_size=0.2, random_state=123)
        val_files, test_files = train_test_split(
            test_val_files, test_size=0.5, random_state=123)

        # Convert file paths to strings
        train_files = [str(file) for file in train_files]
        val_files = [str(file) for file in val_files]
        test_files = [str(file) for file in test_files]
    ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Creating dataframes  {progress_percentage}%'))
    ###############################################################
        # Create dataframes for train, validation, and test files
        train_df = pd.DataFrame({'file_path': train_files, 'label': [
                                os.path.basename(os.path.dirname(f)) for f in train_files]})
        val_df = pd.DataFrame({'file_path': val_files, 'label': [
                            os.path.basename(os.path.dirname(f)) for f in val_files]})
        test_df = pd.DataFrame({'file_path': test_files, 'label': [
                            os.path.basename(os.path.dirname(f)) for f in test_files]})



        class_names = os.listdir(data_dir)
    
        print(class_names)

        # Define the parameters for the model
        
        batch_size = 16


        img_height = 224
        img_width = 224

    ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Augmenting  {progress_percentage}%' ))
    ###############################################################        
    
        train_datagen = ImageDataGenerator(
            # rescale=1.0/255.0,
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            rotation_range=45,             # Adjust rotation range based on your needs
            width_shift_range=0.1,         # Adjust width shift range based on your needs
            height_shift_range=0.1,        # Adjust height shift range based on your needs
            shear_range=0.1,               # Adjust shear range based on your needs
            horizontal_flip=True,          # Horizontal flips can be beneficial
            # vertical_flip=True,  
            fill_mode='reflect',           # Filling strategy for newly created pixels
            brightness_range=[0.8, 1.2],   # Adjust brightness range based on your needs
            zoom_range=[0.8, 1.2],            # Random zoom in/out
            channel_shift_range=10.0,   
        )
        val_datagen = ImageDataGenerator(
            # rescale=1.0/255.0,
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            # preprocessing_function = tf.keras.applications.resnet50.preprocess_input,



        )

        # Create the generator for the train set
        train_ds = train_datagen.flow_from_dataframe(
            train_df,
            x_col='file_path',
            y_col='label',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='sparse',
            seed=123,
            classes=class_names
        )

        # Create the generator for the validation set
        val_ds = val_datagen.flow_from_dataframe(
            val_df,
            x_col='file_path',
            y_col='label',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='sparse',
            seed=123,
            classes=class_names
        )

        # Create the generator for the test set
        test_ds = val_datagen.flow_from_dataframe(
            test_df,
            x_col='file_path',
            y_col='label',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='sparse',
            seed=123,
            classes=class_names
        )

        # # Fine-tune the model for Downy Mildew class
        # downy_mildew_df = train_df[train_df['label'] == 'Downy Mildew']
        # downy_mildew_datagen = ImageDataGenerator(
        #     preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        #     rotation_range=10,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1,
        #     # shear_range=0.1,
        #     horizontal_flip=True,
        #     fill_mode='nearest',
        #     brightness_range=[0.8, 1.2],
        #     zoom_range=[0.8, 1.2],
        #     # featurewise_center=True,
        #     # featurewise_std_normalization=True,
        # )

        # downy_mildew_ds = downy_mildew_datagen.flow_from_dataframe(
        #     downy_mildew_df,
        #     x_col='file_path',
        #     y_col='label',
        #     target_size=(img_height, img_width),
        #     batch_size=batch_size,
        #     class_mode='sparse',
        #     seed=123,
        #     classes=class_names
        # )

                # Create BalancedBatchGenerator for oversampling during training
        # train_ds_resampled = BalancedBatchGenerator(
        #     X_resampled,
        #     y_resampled,
        #     datagen=train_datagen,
        #     batch_size=batch_size,
        #     class_indices=train_df_resampled.class_indices,
        #     random_state=123
        # )

        

    ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Defining architecture  {progress_percentage}%'))
    ###############################################################
        # Define the model architecture using a pre-trained MobileNet model
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(img_height, img_width, 3)
        )
    ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Freezing some layers  {progress_percentage}%'))    
    ###############################################################
        # Freeze some of the pre-trained layers
        base_model.trainable = False
    
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        prediction_layer = tf.keras.layers.Dense(
            len(class_names),
            activation='softmax',
            kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.02)
        )
        # Add dropout to the classification head
        dropout_rate = 0.5  # Adjust the dropout rate as needed

        # model = tf.keras.Sequential([
        #     base_model,
        #     global_average_layer,
        #     Dropout(dropout_rate),
        #     Dense(64, activation='relu'),  # Reduce the number of units
        #     Dropout(dropout_rate),
        #     prediction_layer
        # ])
        

        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            BatchNormalization(),  # Add batch normalization
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            BatchNormalization(),  # Add batch normalization
            Dropout(dropout_rate),
            prediction_layer
        ])

        # Calculate class weights
        # total_samples = len(train_df)
        # class_weights = dict(zip(range(len(class_names)), (total_samples / (len(class_names) * np.bincount(train_df['label'])))))

        # Add learning rate scheduling
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-3,
        #     decay_steps=10000,
        #     decay_rate=0.9)

        # Compile the model with learning rate scheduling
        model.compile(
            # optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            # optimizer='adam',
            optimizer= tf.keras.optimizers.RMSprop(),
             
            # optimizer= tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        

        )

        # # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10)
        
        
    
        ###########################################################
        
        # progress_percentage = int((current_step / total_steps) * 100)
        # with open('status.txt', 'w') as f:
        #         f.write(str(f'Extracting features  {progress_percentage}%'))
        ############################################################
        epoch_status_callback = EpochStatusCallback(total_steps,current_step,epochs)
        current_step +=epochs

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stop, epoch_status_callback],  # Add EpochCountCallback
        
        )
    ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Compiling {progress_percentage}%'))  
    ###############################################################

        
        # prediction_layer = tf.keras.layers.Dense(
        #     len(class_names),
        #     activation='softmax',
        #     kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)
        # )
        # model = tf.keras.Sequential([
        #     base_model,
        #     global_average_layer,
        #     BatchNormalization(),  # Add batch normalization
        #     Dropout(dropout_rate),
        #     Dense(32, activation='relu'),
        #     BatchNormalization(),  # Add batch normalization
        #     Dropout(dropout_rate),
        #     prediction_layer
        # ])
    
        # # # Continue training with a lower learning rate
        # fine_tune_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-3,
        #     decay_steps=10000,
        #     decay_rate=0.9
        # )

        # model.compile(
        #     # optimizer= tf.keras.optimizers.Adam(learning_rate=1e-4),
        #     # optimizer='adam',
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr_schedule),
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        #     metrics=['accuracy'],
        
        # )

        # base_model.trainable = True

        # fine_tune_at = 50

        # for layer in base_model.layers[:fine_tune_at]:
        #     layer.trainable = False

        # # # Continue training
        # # history_fine = model.fit(
        # #     train_ds,
        # #     validation_data=val_ds,
        # #     epochs=epochs,
        # #     initial_epoch=history.epoch[-1],  # Continue training from the last epoch of the initial training
        # #     callbacks=[early_stop],
            
        # # )
    

        # # Continue training the model for Downy Mildew class
        # history_downy_mildew = model.fit(
        #     downy_mildew_ds,
        #     validation_data=val_ds,
        #     epochs=epochs,
        #     callbacks=[early_stop],
        # )

        #SHOW : number of epoch being process in html
        # Save the accuracy plot as an image

        accuracy_img_base64 = create_and_save_plot(
            'Model Accuracy', 'Accuracy', 'Epoch', history.history['accuracy'], history.history['val_accuracy'], epochs)

        # Plot training and validation loss
        loss_img_base64 = create_and_save_plot(
            'Model Loss', 'Loss', 'Epoch', history.history['loss'], history.history['val_loss'], epochs)

        result_data = {
            "success": 'true', 
            "accuracyGraph": accuracy_img_base64,
            "lossGraph": loss_img_base64,
            "loss": history.history['loss'],
            "accuracy": history.history['accuracy'],
            "val_loss": history.history['val_loss'],
            "val_accuracy": history.history['val_accuracy']

        }
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # After training is completed, save the trained model to a .tflite file
        ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Saving {progress_percentage}%'))  
        ###############################################################
        

        # Save the TFLite model to a file
        with open('flask_model/model.tflite', 'wb') as f:
            f.write(tflite_model)

        # Save the class names as a txt file
        with open('flask_model/labels.txt', 'w') as f:
            for class_name in class_names:
                f.write(class_name + '\n')
    ##############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
            f.write(str(f'Done {progress_percentage}%'))  
    ###############################################################
        with open('running_status.txt', 'w') as f:
                f.write(str(f'Not Running'))
        return jsonify(result=result_data)


if __name__ == '__main__':
    app.run(debug=False)
    # Ensure that you pass the eventlet server to the SocketIO.run() method
    # app.run(app)
    # serve(app,host='0.0.0.0', port=5000)
