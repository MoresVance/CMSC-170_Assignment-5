# setup.py
import tensorflow as tf
import json
import requests
import os

def download_and_save_assets():
    print("--- Starting Download and Setup ---")
    
    weights_path = 'models/efficientnetb0.weights.h5'
    labels_path = 'models/imagenet_class_labels.json'
    
    # Check if files already exist
    if os.path.exists(weights_path) and os.path.exists(labels_path):
        print("Setup files already exist!")
        print(f"  - Model weights: {weights_path}")
        print(f"  - Class labels: {labels_path}")
        print("\nSetup has been cancelled. Delete these files if you want to redownload them.")
        print("--- Setup Complete! ---")
        return

    print("Downloading EfficientNetB0 model weights...")
    try:
        model = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=True)
        
        model.save_weights(weights_path)
        print(f"Successfully saved model weights to: {weights_path}")

    except Exception as e:
        print(f"Error downloading or saving model weights: {e}")
        return

    print("\nDownloading ImageNet class labels...")
    try:
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        
        response = requests.get(url)
        response.raise_for_status()
        
        class_index = response.json()
        
        imagenet_classes = [class_index[str(i)][1] for i in range(len(class_index))]
        
        with open(labels_path, 'w') as f:
            json.dump(imagenet_classes, f)
            
        print(f"Successfully saved ImageNet class labels to: {labels_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading class labels: {e}")
    except Exception as e:
        print(f"An error occurred while processing class labels: {e}")

    print("\n--- Setup Complete! ---")

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    
    download_and_save_assets()