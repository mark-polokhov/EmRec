from infer import run_inference
from vit_config import vit_config
from resnet50_config import resnet50_config

from flask import Flask, request, render_template, make_response, jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

import os
import uuid
import subprocess
import pathlib

application = Flask(__name__)

application.config['SECRET_KEY'] = '4441'
application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
application.config['UPLOAD_FOLDER'] = 'static/sessions/'
application.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'gif'}

db = SQLAlchemy(application)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.String(36), nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in application.config['ALLOWED_EXTENSIONS']

@application.route("/", methods=['GET', 'POST'])
@application.route("/main", methods=['GET', 'POST'])
def main():
    user_id = request.cookies.get('user_id')
    if not user_id:
        response = make_response(render_template('main.html'))
        return response
    return render_template('main.html')


@application.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_folder = f"{uuid.uuid4()}"
        while os.path.exists(unique_folder):
            unique_folder = f"{uuid.uuid4()}"
        os.makedirs(os.path.join(application.config['UPLOAD_FOLDER'], unique_folder))
        unique_filename = f"{unique_folder}/{filename}"
        file_path = os.path.join(application.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        try:
            split_result = subprocess.run(['ffmpeg', '-i', file_path, '-r', '20', '-f', 'image2', f'static/sessions/{unique_folder}/image-%07d.png'], 
                                      check=True, stdout=subprocess.PIPE, text=True)
            frames = [file for file in os.listdir(os.path.join(application.config['UPLOAD_FOLDER'], unique_folder)) if file.endswith('.png')]
            full_path = str(pathlib.Path(__file__).parent.resolve()).replace('\\', '/') + '/' + application.config['UPLOAD_FOLDER'] + unique_folder
            subprocess.run(f'python3 infer.py {" ".join(vit_config + [f"-d {full_path}"] + [f"-o {full_path}/predicts_vit.txt"])}', shell=True)
            subprocess.run(f'python3 infer.py {" ".join(resnet50_config + [f"-d {full_path}"] + [f"-o {full_path}/predicts_resnet.txt"])}', shell=True)
            with open(f"{full_path}/predicts_vit.txt", 'r') as predicts_file:
                labels = predicts_file.readline().split()
                predicts_vit = [list(map(lambda x : round(float(x) * 100, 1), row.strip('\n').split())) for row in predicts_file.readlines()]
            with open(f"{full_path}/predicts_resnet.txt", 'r') as predicts_file:
                labels = predicts_file.readline().split()
                predicts_resnet = [list(map(lambda x : round(float(x) * 100, 1), row.strip('\n').split())) for row in predicts_file.readlines()]
            return jsonify({'success': True, 'output': split_result.stdout, 'uuid': unique_folder, 'num_frames': len(frames),
                            'classes': labels, 'predicts_vit': predicts_vit, 'predicts_resnet': predicts_resnet})
        except subprocess.CalledProcessError as e:
            return jsonify({'success': False, 'error': str(e)})
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid file type'})

@application.route('/cleanup', methods=['POST'])
def cleanup():
    session_id = request.get_data(as_text=True)
    delete_files(session_id)
    return '', 204

def delete_files(session_id):
    full_path = str(pathlib.Path(__file__).parent.resolve()).replace('\\', '/') + '/' + application.config['UPLOAD_FOLDER'] + session_id
    for filename in os.listdir(full_path):
        os.remove(full_path + '/' + filename)
    os.rmdir(full_path)
    return

if __name__ == "__main__":
    # Ensure the upload folder exists
    if not os.path.exists(application.config['UPLOAD_FOLDER']):
        os.makedirs(application.config['UPLOAD_FOLDER'])
    
    with application.app_context():
        db.create_all()
    application.run(host='0.0.0.0', port=8080)