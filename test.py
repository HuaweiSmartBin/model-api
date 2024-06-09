import requests

url = "http://127.0.0.1:8000/predict"
file_path = r""  # Path to the image file

with open(file_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, files=files)

print(response.json())