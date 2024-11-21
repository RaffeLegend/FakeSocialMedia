import os
from PIL import Image

def load_data(data_type, data_path):
    """
    Load data based on the specified type.

    Parameters:
    data_type (str): Type of data to load ('image', 'text', 'image_text_pair').
    data_path (str): Path to the data directory.

    Returns:
    list: Loaded data.
    """
    data = []

    if data_type == 'fake_image':
        for file_name in os.listdir(data_path):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                # image = Image.open(os.path.join(data_path, file_name))
                data.append((image, os.path.join(data_path, file_name)))

    elif data_type == 'real_image':
        for file_name in os.listdir(data_path):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(os.path.join(data_path, file_name))
                data.append(image)

    elif data_type == 'real_fake_image':
        for file_name in os.listdir(data_path):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(os.path.join(data_path, file_name))
                data.append(image)
    
    elif data_type == 'text':
        for file_name in os.listdir(data_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(data_path, file_name), 'r') as file:
                    text = file.read()
                    data.append(text)
    
    elif data_type == 'image_text_pair':
        for file_name in os.listdir(data_path):
            if file_name.endswith('.txt'):
                text_file_path = os.path.join(data_path, file_name)
                image_file_path = text_file_path.replace('.txt', '.jpg')
                if os.path.exists(image_file_path):
                    with open(text_file_path, 'r') as file:
                        text = file.read()
                    image = Image.open(image_file_path)
                    data.append((image, text))
    
    else:
        raise ValueError("Unsupported data type. Choose from 'image', 'text', 'image_text_pair'.")

    return data