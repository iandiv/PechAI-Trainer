# from urllib import request
from flask import Flask, request, render_template, jsonify
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import base64
import io
import tensorflow as tf
import pathlib
import os
import numpy as np
import pandas as pd

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from PIL import Image
import numpy as np
import tensorflow.lite as tflite
import matplotlib
matplotlib.use('agg')  # Set Matplotlib to use the 'agg' backend


api_url = "http://localhost/PechAI/img-api.php"

# Make a GET request to the API
response = request.get(api_url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Assuming the API returns JSON data as shown in your example

    # Create dataframes for train, validation, and test files
    train_df = pd.DataFrame(columns=['file_path', 'label'])

    # Iterate through the API data to populate the dataframes
    for category, image_urls in data.items():
        for image_url in image_urls:
            # Append the image URL and category label to the dataframe
            train_df = train_df.append({'file_path': image_url, 'label': category}, ignore_index=True)

    # Define the classes (diseases) that we want to detect
    class_names = list(data.keys())
    class_names.append("Invalid")  # Add "Invalid" class name

    # Print the list of class names
    print(class_names)

else:
    print(f"Failed to retrieve data from the API. Status code: {response.status_code}")
