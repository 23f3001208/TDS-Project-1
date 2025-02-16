import sqlite3
import subprocess
from dateutil.parser import parse
import json
from pathlib import Path
import os
from fastapi import HTTPException
import requests
from dotenv import load_dotenv
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance

load_dotenv()

AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')

def A1(email="23f3001208@ds.study.iitm.ac.in"):
    try:
        process = subprocess.Popen(
            ["uv", "run", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", email],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error: {stderr}")
        return stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")

def A2(prettier_version="prettier@3.4.2", filename="/data/format.md"):
    command = [r"C:\Program Files\nodejs\npx.cmd", prettier_version, "--write", filename]
    try:
        subprocess.run(command, check=True)
        print("Prettier executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def A3(filename='/data/dates.txt', targetfile='/data/dates-wednesdays.txt', weekday=2):
    input_file = filename
    output_file = targetfile
    weekday = weekday
    weekday_count = 0

    with open(input_file, 'r') as file:
        weekday_count = sum(1 for date in file if parse(date).weekday() == int(weekday)-1)


    with open(output_file, 'w') as file:
        file.write(str(weekday_count))

def A4(filename="/data/contacts.json", targetfile="/data/contacts-sorted.json"):
    # Load the contacts from the JSON file
    with open(filename, 'r') as file:
        contacts = json.load(file)

    # Sort the contacts by last_name and then by first_name
    sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

    # Write the sorted contacts to the new JSON file
    with open(targetfile, 'w') as file:
        json.dump(sorted_contacts, file, indent=4)

def A5(log_dir_path='/data/logs', output_file_path='/data/logs-recent.txt', num_files=10):
    log_dir = Path(log_dir_path)
    output_file = Path(output_file_path)

    # Get list of .log files sorted by modification time (most recent first)
    log_files = sorted(log_dir.glob('*.log'), key=os.path.getmtime, reverse=True)[:num_files]

    # Read first line of each file and write to the output file
    with output_file.open('w') as f_out:
        for log_file in log_files:
            with log_file.open('r') as f_in:
                first_line = f_in.readline().strip()
                f_out.write(f"{first_line}\n")

def A6(doc_dir_path='/data/docs', output_file_path='/data/docs/index.json'):
    docs_dir = doc_dir_path
    output_file = output_file_path
    index_data = {}

    # Walk through all files in the docs directory
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                # print(file)
                file_path = os.path.join(root, file)
                # Read the file and find the first occurrence of an H1
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('# '):
                            # Extract the title text after '# '
                            title = line[2:].strip()
                            # Get the relative path without the prefix
                            relative_path = os.path.relpath(file_path, docs_dir).replace('\\', '/')
                            index_data[relative_path] = title
                            break  # Stop after the first H1
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4)

def A7(filename='/data/email.txt', output_file='/data/email-sender.txt'):
    # Read the content of the email
    with open(filename, 'r') as file:
        email_content = file.readlines()

    sender_email = "sujay@gmail.com"
    for line in email_content:
        if "From" == line[:4]:
            sender_email = (line.strip().split(" ")[-1]).replace("<", "").replace(">", "")
            break

    with open(output_file, 'w') as file:
        file.write(sender_email)

# Function to preprocess the image
def preprocess_image(image_path, output_path):
    # Load image using Pillow and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Slightly enhance contrast and sharpness to avoid excessive thickening
    image = ImageEnhance.Contrast(image).enhance(1.1)  # Reduce contrast enhancement
    image = ImageEnhance.Sharpness(image).enhance(1.1)  # Reduce sharpness enhancement

    # Convert to OpenCV format for further processing
    image = np.array(image)

    # Apply bilateral filter to remove noise but keep edges sharp
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Upscale once with cubic interpolation
    scale_factor = 2.0  # Single upscale to avoid excessive boldening
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    # Apply Otsu's thresholding for better digit separation
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological thinning to reduce digit thickness
    kernel = np.ones((1, 1), np.uint8)  # Small kernel to prevent over-thinning
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

    # Save preprocessed image
    cv2.imwrite(output_path, image)
    print(f"✅ Preprocessed image saved at: {output_path}")
    return output_path

def png_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

def A8(filename='/data/credit_card.txt', image_path='/data/credit_card.png'):
    # Construct the request body for the AIProxy call
    processed_image = preprocess_image(image_path, '/data/processed_credit_card.png')  # Preprocess before sending

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "There is 8 or more digit number is there in this image, with space after every 4 digit, only extract the those digit number without spaces and return just the number without any other characters"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{png_to_base64(processed_image)}"
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    # Make the request to the AIProxy service
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                             headers=headers, data=json.dumps(body))
    # response.raise_for_status()

    # Extract the credit card number from the response
    result = response.json()
    # print(result); return None
    card_number = result['choices'][0]['message']['content'].replace(" ", "")

    # Write the extracted card number to the output file
    with open(filename, 'w') as file:
        file.write(card_number)

def A9(filename='/data/comments.txt', output_filename='/data/comments-similar.txt'):

    with open(filename, "r") as f:
        comments = [line.strip() for line in f.readlines() if line.strip()]

    # Request embeddings in a batch for efficiency
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": comments}
    
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/embeddings", json=payload, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch embeddings: {response.text}")

    embeddings = response.json().get("data", [])
    if len(embeddings) != len(comments):
        raise ValueError("Mismatch between comments and embeddings count.")

    # Compute cosine similarity
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    max_similarity = -1
    best_pair = ("", "")

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            similarity = cosine_similarity(embeddings[i]["embedding"], embeddings[j]["embedding"])
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (comments[i], comments[j])

    with open(output_filename, "w") as f:
        f.write(f"{best_pair[0]}\n{best_pair[1]}")

def A10(filename='/data/ticket-sales.db', output_filename='/data/ticket-sales-gold.txt', query="SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"):
    # Connect to the SQLite database
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()

    # Calculate the total sales for the "Gold" ticket type
    cursor.execute(query)
    total_sales = cursor.fetchone()[0]

    # If there are no sales, set total_sales to 0
    total_sales = total_sales if total_sales else 0

    # Write the total sales to the file
    with open(output_filename, 'w') as file:
        file.write(str(total_sales))

    # Close the database connection
    conn.close()