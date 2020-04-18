import os
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.secret_key = "SECRET_KEY"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #define max size of input file here

ALLOWED_EXTENSIONS = set(['mp4', 'avi', 'mkv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.errorhandler(413)
# def request_entity_too_large(error):
#     flash('Allowed file size is 16MB')
#     return redirect('request.url'), 413

@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Successfully Uploaded')
            return redirect('/')
        else:
            flash('Allowed file type is mp4')
            return redirect(request.url)

    
if __name__ == "__main__":
    app.run(debug=True)