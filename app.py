from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import uuid
import numpy as np
app = Flask(__name__)

# Define the upload folder 
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
MODEL_PATH = "skin_unknown.pth"
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

# Define the class labels
CLASS_LABELS = {
    0: "Actinic keratoses and intraepithelial carcinoma ",
    1: "Basal cell carcinoma ",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma ",
    4: "Melanoma ",
    5: "Melanocytic nevi ",
    6: "Unknown",
    7: "Vascular lesions"
}

# Malignancy status for each class
MALIGNANCY_STATUS = {
    0: "Malignant",
    1: "Malignant",
    2: "Benign",
    3: "Benign",
    4: "Malignant",
    5: "Benign",
    6: "N/A",
    7: "Benign"
}

# Recommendations for each malignancy status
RECOMMENDATIONS = {
    "Malignant": "Visit a dermatologist immediately for further diagnosis and treatment.",
    "Benign": "The lesion appears benign. However, monitor for changes and consult a dermatologist if concerned.",
    "Unknown": "The Image is not recognized. The uploaded image may not be a valid skin lesion."
}

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Generate a unique filename for the uploaded image
            unique_filename = str(uuid.uuid4()) + ".jpg"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

            # Save the uploaded file
            file.save(file_path)

            # Process the uploaded image
            img = Image.open(file_path).convert('RGB')
            img = transform(img).unsqueeze(0)

            # Predict using the model
            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                class_index = predicted.item()

            class_label = CLASS_LABELS.get(class_index, "Unknown")
            malignancy_status = MALIGNANCY_STATUS.get(class_index, "Unknown")
            recommendation = RECOMMENDATIONS.get(malignancy_status, "No recommendation available.")

            return jsonify({
                "class": class_label,
                "malignancy_status": malignancy_status,
                "confidence": f"{torch.softmax(outputs, 1)[0][class_index].item() * 100:.2f}%",
                "recommendation": recommendation,
                "image_path": f"/static/uploads/{unique_filename}"
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid request"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
