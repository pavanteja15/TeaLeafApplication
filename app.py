from flask import Flask, render_template, request, redirect, flash, url_for
import os
import shutil
from ultralytics import YOLO 

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load a pretrained YOLOv8n model
model = YOLO("best.pt")

UPLOAD_FOLDER = 'uploads'
PREDICT_FOLDER = 'static/predict'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_folders():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(PREDICT_FOLDER):
        os.makedirs(PREDICT_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    create_folders()

    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Predict using YOLO model and save results
        model.predict(filename, save=True, save_txt=True, hide_conf=True)

        # Get the latest prediction image from `runs/classify/predict` folder
        image_files = [f for f in os.listdir('runs/classify/predict') if f.endswith(('.jpg', '.png'))]
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join('runs/classify/predict', x)), reverse=True)

        if image_files:
            latest_image_path = os.path.join('runs/classify/predict', image_files[0])
            dest_image_path = os.path.join(PREDICT_FOLDER, image_files[0])
            
            # Copy the image to static/predict for Flask to serve
            shutil.copy(latest_image_path, dest_image_path)

            # Generate the image URL for rendering in the template
            image_url = url_for('static', filename=f'predict/{image_files[0]}')

            # Read the prediction label from the labels file
            txt_files = os.listdir('runs/classify/predict/labels')
            txt_files.sort(key=lambda x: os.path.getmtime(os.path.join('runs/classify/predict/labels', x)), reverse=True)

            if txt_files:
                txt_file_path = os.path.join('runs/classify/predict/labels', txt_files[0])
                with open(txt_file_path, 'r') as txt_file:
                    first_line = txt_file.readline()[4:].strip()

                result = "The tea leaf is healthy!" if first_line.lower() == 'healthy' else f'The given leaf has: {first_line} Disease'
                
                return render_template('result.html', result=result, image_url=image_url)

    flash('No files found in the "labels" directory.', 'error')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
