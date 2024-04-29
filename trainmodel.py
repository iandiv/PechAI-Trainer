# Standard Library Imports
import os
import json
import sys
import base64
import io
import pathlib
import shutil
import signal
import subprocess
import random
import concurrent.futures
import time

# External Library Imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.lite as tflite
import requests
from PIL import Image
from sklearn.utils import resample
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras.applications import MobileNetV2

# Other Configurations
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
tf.config.optimizer.set_jit(True)
matplotlib.use('agg')  # Set Matplotlib to use the 'agg' backend


# Create a Flask web application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_upload'
# app.config['STATIC_FOLDER'] = 'static'
# Auto apply UI changes 
app.config['TEMPLATES_AUTO_RELOAD'] = True 

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


# def download_image(image_url, output_dir):
#     image_filename = os.path.join(output_dir, os.path.basename(image_url))
#     if not os.path.exists(image_filename):
#         image_response = requests.get(image_url)
#         if image_response.status_code == 200:
#             with open(image_filename, 'wb') as image_file:
#                 image_file.write(image_response.content)
#             return "Downloaded: " + image_filename
#         else:
#             return "Failed to download: " + image_url
#     else:
#         return "Already exists: " + image_filename

# def download_images_from_api(api_url, api_key, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     response = requests.post(api_url, data={'api_key': api_key})

#     if response.status_code == 200:
#         data = json.loads(response.text)
#         all_image_urls = []

#         # Gather all image URLs from the API response
#         for label, image_urls in data.items():
#             label_dir = os.path.join(output_dir, label)
#             if not os.path.exists(label_dir):
#                 os.makedirs(label_dir)
#             all_image_urls.extend([(url, label_dir) for url in image_urls])

#         # Download images concurrently
#         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#             results = list(executor.map(lambda item: download_image(*item), all_image_urls))

#         return results
#     else:
#         return f"Error: Request failed with status code {response.status_code}"

downloaded_images = {}



def download_image(session, image_url, output_dir):
    # Check if the image URL is already downloaded
    if image_url in downloaded_images:
        return f"Already exists: {downloaded_images[image_url]}"

    image_filename = os.path.join(output_dir, os.path.basename(image_url))
    if not os.path.exists(image_filename):
        image_response = session.get(image_url, stream=True)
        if image_response.status_code == 200:
            with open(image_filename, 'wb') as image_file:
                for chunk in image_response.iter_content(1024):
                    image_file.write(chunk)
            # Update the downloaded_images dictionary
            downloaded_images[image_url] = image_filename
            return f"Downloaded: {image_filename}"
        else:
            return f"Failed to download: {image_url}"
    else:
        # Update the downloaded_images dictionary
        downloaded_images[image_url] = image_filename
        return f"Already exists: {image_filename}"

def download_images_from_api(api_url, api_key, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    session = requests.Session()  # Create a session for persistent connections
    response = session.post(api_url, data={'api_key': api_key})

    if response.status_code == 200:
        data = json.loads(response.text)
        all_image_urls = []

        # Gather all image URLs from the API response
        for label, image_urls in data.items():
            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            all_image_urls.extend([(image_url, label_dir) for image_url in image_urls])

        # Download images concurrently with a thread pool executor
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda item: download_image(session, *item), all_image_urls))

        return results
    else:
        return f"Error: Request failed with status code {response.status_code}"










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

# Custom callback to save metrics after each epoch as arrays
class SaveMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.accuracy = []
        self.val_accuracy = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Add the current epoch's metrics to the respective lists
        self.accuracy.append(logs.get("accuracy", 0.0))
        self.val_accuracy.append(logs.get("val_accuracy", 0.0))
        self.loss.append(logs.get("loss", 0.0))
        self.val_loss.append(logs.get("val_loss", 0.0))

        accuracy_img_base64 = create_and_save_plot(
            'Model Accuracy', 'Accuracy', 'Epoch',self.accuracy, self.val_accuracy, epoch)

        # Plot training and validation loss
        loss_img_base64 = create_and_save_plot(
            'Model Loss', 'Loss', 'Epoch',self.loss,  self.val_loss, epoch)

        # Create the output dictionary with arrays of metric values
        output_data = {
            "result": {
                "success": 'true',
                "accuracyGraph": accuracy_img_base64,
                "lossGraph": loss_img_base64,
                "accuracy": self.accuracy,
                "loss": self.loss,
                "val_accuracy": self.val_accuracy,
                "val_loss": self.val_loss
            }
        }

        # Save the dictionary to a JSON file
        with open(self.file_path, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)

        print(f"Epoch {epoch + 1} metrics saved to {self.file_path}")


@app.route('/')
def index():
    with open('running_status.txt', 'r') as f:
        content = f.read()

        if "Stopping" in content:
            with open('running_status.txt', 'w') as f:
                f.write(str(f'Not Running'))
            with open('status.txt', 'w') as f:
                f.write(str(f'Initializing 0%'))
    return render_template('train.html')


@app.route('/results')
def results():
    with open('results.json', 'r') as f:
        content = f.read()
    return content

@app.route('/status')
def status():
    with open('status.txt', 'r') as f:
        content = f.read()
    return content

@app.route('/running_status')
def running_status():
    with open('running_status.txt', 'r') as f:
        content = f.read()
    return content





# Function to stop a running process by its process ID
def stop_process(pid):
    try:
        # Send SIGTERM to stop gracefully
        os.kill(pid, signal.SIGTERM)
        # Wait a bit for the process to stop
        time.sleep(2)

        # If the process is still running, force it to stop
        if os.getpgid(pid) == pid:
            os.kill(pid, signal.SIGKILL)
    except Exception as e:
        print(f"Error stopping process: {e}")

# Function to start a new process
def start_new_process(script_path):
    subprocess.Popen(['powershell', '-File', script_path])



@app.route('/restart', methods=['POST'])
def restart():
    # process = subprocess.Popen(['powershell', 'run_trainmodel.ps1'])
    
    
    # Update status files
    with open('running_status.txt', 'w') as f:
        f.write('Stopping')
    with open('status.txt', 'w') as f:
        f.write('Stopping')

    # Restart the Flask server (may require process manager to complete successfully)
    # os.kill(os.getpid(), signal.SIGTERM)
    # Start the virtual environment and run trainmodel.py
    # os.kill(os.getpid(), signal.SIGINT)
    # subprocess.Popen(['powershell', 'run_trainmodel.ps1'])

    # Stop the current process
    

    # Start a new instance of the PowerShell script
    start_new_process('run_trainmodel.ps1')
    current_pid = os.getpid()

    stop_process(current_pid)
    # os.execv(sys.executable, [sys.executable] + sys.argv)
    
    return 'Restarting Flask server'






@app.route('/upload', methods=[ 'POST'])
def upload():
  if request.method == 'POST':
        api_url = 'https://pechai.site/model-api.php'

        labels_file_path = 'flask_model/labels.txt'
        model_file_path = 'flask_model/model.tflite'

        labels_base64 = encode_file_to_base64(labels_file_path)
        model_base64 = encode_file_to_base64(model_file_path)

        data = {
            'labels': labels_base64,
            'model': model_base64
        }

        # Send POST request to the API endpoint with files
        response = requests.post(api_url, data=data)

        # Print the response from the server
        return response.text

    

@app.route('/test-result', methods=['POST'])
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
        input_image = np.array(image, dtype=np.float32)  / 255.0
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



@app.route('/train_model', methods=['POST'])
def train_model():
        # Your data dictionary
    with open('status.txt', 'w') as f:
         f.write(str(f'...'))
   
    # Read the JSON file
    with open('results.json', 'r') as file:
        data = json.load(file)

    # Update the value of "success" to False
    data["result"]["success"] = "false"

    # Write the updated data back to the JSON file
    with open('results.json', 'w') as file:
        json.dump(data, file)
    preprocessing = {
        MobileNetV2: tf.keras.applications.mobilenet_v2.preprocess_input,
    }
    arhictecture = MobileNetV2

    with open('running_status.txt', 'w') as f:
                f.write(str(f'Running'))
    epochs = 50
    total_steps = 10+epochs 
    current_step = 0  #

    


    ###############################################################
    current_step += 1
    progress_percentage = int((current_step / total_steps) * 100)
    with open('status.txt', 'w') as f:
                f.write(str(f'Initializing {progress_percentage}%'))
    ###############################################################
    data_dir = 'datasets'  # Replace with your dataset directory
    # success, message = delete_datasets_folder(data_dir)
    # if success:
    #     print(message)
    # else:
    #     print(f"Error: {message}")

    #SHOW : deleted in html
    # Call the function to download images from the API
    api_url = 'https://pechai.site/img-api.php'
    api_key = request.form.get('apiKey')
    response = requests.get(api_url, verify=False)
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
    
    if any('Error' in resul for resul in result):
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

        ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Splitting datasets  {progress_percentage}%' ))
    ###############################################################        
        train_size = 80 
        val_size = 10
        test_size = 10
        data_dir = pathlib.Path(data_dir)

        all_files = list(data_dir.glob('*/*.jpg'))

        random.shuffle(all_files) 

        total_files = len(all_files)


        # Calculate the number of files for each split based on the specified percentages
        train_num = int(total_files * (train_size / 100))
        val_num = int(total_files * (val_size / 100))
        test_num = int(total_files * (test_size / 100))

        # Split the data into train and remaining
        train_files, remaining_files = train_test_split(all_files, train_size=train_num, stratify=[os.path.basename(os.path.dirname(f)) for f in all_files])

        # Split the remaining data into validation and test sets
        val_files, test_files = train_test_split(remaining_files, test_size=(test_num / (val_num + test_num)), stratify=[os.path.basename(os.path.dirname(f)) for f in remaining_files])

        # Convert file paths to strings
        train_files = [str(file) for file in train_files]
        val_files = [str(file) for file in val_files]
        test_files = [str(file) for file in test_files]

        # Create dataframes for train, validation, and test files
        train_df = pd.DataFrame({'file_path': train_files, 'label': [os.path.basename(os.path.dirname(f)) for f in train_files]})
        val_df = pd.DataFrame({'file_path': val_files, 'label': [os.path.basename(os.path.dirname(f)) for f in val_files]})
        test_df = pd.DataFrame({'file_path': test_files, 'label': [os.path.basename(os.path.dirname(f)) for f in test_files]})




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
            preprocessing_function = preprocessing[arhictecture],
            
            # Apply augmentation
            rotation_range=5,             # Adjust rotation range based on your needs
            width_shift_range=0.1,         # Adjust width shift range based on your needs
            height_shift_range=0.1,        # Adjust height shift range based on your needs
            shear_range=0.1,               # Adjust shear range based on your needs
            horizontal_flip=True,    
            
            vertical_flip=True,        
            brightness_range=[0.5, 1.5],  
            zoom_range=[0.5, 1.2],          
        )

        val_datagen = ImageDataGenerator(
            preprocessing_function = preprocessing[arhictecture]
        )


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


        val_ds = val_datagen.flow_from_dataframe(
            val_df,
            x_col='file_path',
            y_col='label',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='sparse',
            seed=123,
            classes=class_names,
            shuffle = False
        )


        test_ds = val_datagen.flow_from_dataframe(
            test_df,
            x_col='file_path',
            y_col='label',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='sparse',
            seed=123,
            classes=class_names,
            shuffle = False
        )


        

    ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Defining architecture  {progress_percentage}%'))
    ###############################################################
        # Define the model architecture using a pre-trained MobileNet model
        base_model = arhictecture(
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
        )


        dropout_rate = 0.1


        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(16, activation='relu',   kernel_regularizer=regularizers.l1_l2(l1=1e-6)),
            BatchNormalization(), 
            Dropout(dropout_rate),
            prediction_layer
        ])

        # Compile the model with learning rate scheduling
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(),
   
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        

        )

        # # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5)
        
        
    
        
        epoch_status_callback = EpochStatusCallback(total_steps,current_step,epochs)
        current_step +=epochs
        save_metrics_callback = SaveMetricsCallback('results.json')


        lr_reduce_val = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # Monitor validation loss
            factor=0.9,          # Reduce learning rate by half
            min_delta=0.001,     # Minimum change to qualify as an improvement
            patience=1,          # Number of epochs with no improvement before reducing LR
            cooldown=1,          # Cooldown period after LR reduction before resuming normal operation
            min_lr=1e-9,         # Minimum learning rate
            verbose=1            # Show messages
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[lr_reduce_val, epoch_status_callback,save_metrics_callback],  # Add EpochCountCallback
        

        )


     
    ###############################################################
        current_step += 1
        progress_percentage = int((current_step / total_steps) * 100)
        with open('status.txt', 'w') as f:
                f.write(str(f'Compiling {progress_percentage}%'))  
    ###############################################################

    
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
            f.write(str(f'Done 100%'))  
    ###############################################################
        with open('running_status.txt', 'w') as f:
                f.write(str(f'Not Running'))

        output_data = {
        "result": result_data
        }
        # Save the dictionary to a JSON file
        with open("results.json", "w") as json_file:
            json.dump(output_data, json_file)
        with open('status.txt', 'w') as f:
            f.write(str(f''))
        return jsonify(result=result_data)


if __name__ == '__main__':
    app.run(debug=False)
    # Ensure that you pass the eventlet server to the SocketIO.run() method
    # app.run(app)
    # serve(app,host='0.0.0.0', port=5000)
