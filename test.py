
import requests
import os
import json
from flask import Flask, jsonify, request, render_template
import shutil


def download_image(image_url, output_dir):
    image_filename = os.path.join(output_dir, os.path.basename(image_url))
    if not os.path.exists(image_filename):
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            with open(image_filename, 'wb') as image_file:
                image_file.write(image_response.content)
            return "Downloaded: " + image_filename
        else:
            return "Failed to download: " + image_url
    else:
        return "Already exists: " + image_filename

def download_images_from_api(api_url, api_key, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    response = requests.post(api_url, data={'api_key': api_key})

    if response.status_code == 200:
        data = json.loads(response.text)
        all_image_urls = []

        # Gather all image URLs from the API response
        for label, image_urls in data.items():
            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            all_image_urls.extend([(url, label_dir) for url in image_urls])

        # Download images concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda item: download_image(*item), all_image_urls))

        return results
    else:
        return f"Error: Request failed with status code {response.status_code}"


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


data_dir = 'datasets'  # Replace with your dataset directory
success, message = delete_datasets_folder(data_dir)

api_url = 'https://pechai.site/img-api.php'
api_key = "MtMmLomDuTNSOYvgVVVHDnaf17zsQ0Av"
response = requests.get(api_url, verify=False)


result = download_images_from_api(api_url, api_key, data_dir)