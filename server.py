"""Python Flask WebApp Auth0 integration example
"""
import glob
import json
import logging
import os
import time
import urllib.request
from functools import wraps
from os import environ as env

import cv2
import dlib
import face_recognition
import numpy as np
from authlib.integrations.flask_client import OAuth
from celery import Celery, group
from dotenv import find_dotenv, load_dotenv
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from flask_sqlalchemy import SQLAlchemy
from imutils import face_utils
from six.moves.urllib.parse import urlencode
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import constants
from notebooks.preprocess import preprocess_input
from notebooks.utils import *
import ray
ray.init(num_cpus=4, ignore_reinit_error=True)

logging.basicConfig(level=logging.DEBUG)

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

AUTH0_CALLBACK_URL = env.get(constants.AUTH0_CALLBACK_URL)
AUTH0_CLIENT_ID = env.get(constants.AUTH0_CLIENT_ID)
AUTH0_CLIENT_SECRET = env.get(constants.AUTH0_CLIENT_SECRET)
AUTH0_DOMAIN = env.get(constants.AUTH0_DOMAIN)
AUTH0_BASE_URL = 'https://' + AUTH0_DOMAIN
AUTH0_AUDIENCE = env.get(constants.AUTH0_AUDIENCE)
UPLOAD_FOLDER = './public/'
ALLOWED_EXTENSIONS = set(['mp4', 'avi', 'mkv'])


app = Flask(__name__, static_url_path='/public', static_folder='./public')
app.secret_key = constants.SECRET_KEY
app.debug = True
app.config.from_object("config.Config")

db = SQLAlchemy(app)

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(128), unique=True, nullable=False)
    active = db.Column(db.Boolean(), default=True, nullable=False)

    def __init__(self, email):
        self.email = email

class Video(db.Model):
    __tablename__ = "videos"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    video_path = db.Column(db.String(128))
    video_title = db.Column(db.String(128))
    processed = db.Column(db.Boolean, default=False)

    def __init__(self, user_id, vid_title):
        self.user_id = user_id
        self.video_title = vid_title

from tasks import create_user, create_vid, add_vid_path

oauth = OAuth(app)

auth0 = oauth.register(
    'auth0',
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    api_base_url=AUTH0_BASE_URL,
    access_token_url=AUTH0_BASE_URL + '/oauth/token',
    authorize_url=AUTH0_BASE_URL + '/authorize',
    client_kwargs={
        'scope': 'openid profile email',
    },
)


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if constants.PROFILE_KEY not in session:
            return redirect('/login')
        return f(*args, **kwargs)

    return decorated


# Controllers API
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/callback')
def callback_handling():
    auth0.authorize_access_token()
    resp = auth0.get('userinfo')
    userinfo = resp.json()

    session[constants.JWT_PAYLOAD] = userinfo
    print(userinfo)
    session[constants.PROFILE_KEY] = {
        'user_id': userinfo['sub'],
        'name': userinfo['name'],
        'picture': userinfo['picture']
    }
    return redirect(url_for('dashboard'))


@app.route('/login')
def login():
    return auth0.authorize_redirect(redirect_uri=AUTH0_CALLBACK_URL, audience=AUTH0_AUDIENCE)


@app.route('/logout')
def logout():
    session.clear()
    params = {'returnTo': url_for('home', _external=True), 'client_id': AUTH0_CLIENT_ID}
    return redirect(auth0.api_base_url + '/v2/logout?' + urlencode(params))

@celery.task(bind=True)
def create_user(self, email):
    new_user = User(email=email)
    db.session.add(new_user)
    db.session.commit()
    self.update_state(state='COMPLETED')
    logging.info("Created")
    return 1

@celery.task(bind=True)
def create_vid(self, user_id, vid_name):
    vid = Video(user_id, vid_name)
    db.session.add(vid)
    db.session.commit()

@celery.task(bind=True)
def add_vid_path(self, video_id):
    vid = Video.query.filter_by(id=video_id).first()
    vid.video_path = str(vid.user_id) + "/" + str(video_id)
    db.session.add(vid)
    db.session.commit()

def generate_emotion_video(ray_list, vid_path, size):
    cap = cv2.VideoCapture(vid_path)
    logging.info(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # while cap.isOpened():
    #     ret,frame = cap.read()
    #     if frame is None:
    #         break
    #     height, width, layers = frame.shape
    #     size = (width,height)
    #     break
    cap.release()
    out = cv2.VideoWriter(vid_path + '_emotion.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    ray_list = ray.get(ray_list)
    for iterx in ray_list:
        out.write(iterx[1])

    out.release()
    logging.info(vid_path[len(UPLOAD_FOLDER) + 1:])
    video = Video.query.filter_by(video_path=vid_path[len(UPLOAD_FOLDER) + 1:]).first()
    video.processed = True
    db.session.commit()
    return 1

@ray.remote
class Model(object):

	def __init__(self):
		from keras.models import load_model
		emotion_model_path = './notebooks/model.hdf5'
		self.labels = {
			0:'angry',
			1:'disgust',
			2:'fear',
			3:'happy',
			4:'sad',
			5:'surprise',
			6:'neutral'
		}
		self.frame_window = 10
		self.emotion_offsets = (20, 40)
		self.detector = dlib.get_frontal_face_detector()
		self.emotion_classifier = load_model(emotion_model_path)
	

	def predictFace(self, gray_image, face):
		emotion_target_size = self.emotion_classifier.input_shape[1:3]

		x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face), self.emotion_offsets)
		gray_face = gray_image[y1:y2, x1:x2]

		try:
			gray_face = cv2.resize(gray_face, (emotion_target_size))
		except:
			return None
		gray_face = preprocess_input(gray_face, True)
		gray_face = np.expand_dims(gray_face, 0)
		gray_face = np.expand_dims(gray_face, -1)
		emotion_prediction = self.emotion_classifier.predict(gray_face)
		emotion_probability = np.max(emotion_prediction)
		emotion_label_arg = np.argmax(emotion_prediction)
		emotion_text = self.labels[emotion_label_arg]
		return emotion_text

	def predictFrame(self, frame):
		gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		faces = self.detector(rgb_image)
		each_face_emotion = []
		for face in faces:
			each_face_emotion.append(self.predictFace(gray_image, face))

			if each_face_emotion[-1] == 'angry':
				color = np.asarray((255, 0, 0))
			elif each_face_emotion[-1] == 'sad':
				color = np.asarray((0, 0, 255))
			elif each_face_emotion[-1] == 'happy':
				color = np.asarray((255, 255, 0))
			elif each_face_emotion[-1] == 'surprise':
				color = np.asarray((0, 255, 255))
			elif each_face_emotion[-1] == 'disgust':
				color = np.asarray((0, 255, 0))				
			else:
				color = np.asarray((255, 255, 255))

			color = color.astype(int)
			color = color.tolist()

			name = each_face_emotion[-1]
			
			draw_bounding_box(face_utils.rect_to_bb(face), rgb_image, color)
			draw_text(face_utils.rect_to_bb(face), rgb_image, name,color, 0, -45, 0.5, 1)
		
		tframe = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		return each_face_emotion,tframe

@ray.remote
def process_vid(vid_path):
    logging.info("Process Vid")
    logging.info(vid_path)
    cap = cv2.VideoCapture(vid_path)
    all_emotions = []
    detect = Model.remote()
    frames = []
    size = None
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        height, width, layers = frame.shape
        size = (width, height)

        
        all_emotions.append(detect.predictFrame.remote(frame))
    task = generate_emotion_video(all_emotions, vid_path, size)
    logging.info(task)
    
celery.register_task(create_user)
celery.register_task(create_vid)
celery.register_task(add_vid_path)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/process_vid/<int:user_id>/<int:video_id>")
def vid(user_id, video_id):
    logging.info("In Function")
    vid_path = os.path.join(app.config["UPLOAD_FOLDER"], str(user_id), str(video_id))
    logging.info(vid_path)
    start = time.time()
    task = process_vid.remote(vid_path)
    logging.info(task)
    end = time.time()
    logging.info(end - start)
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@requires_auth
def dashboard():
    user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first()
    logging.info(user)
    if user is None:
        # task = create_user.apply(session[constants.JWT_PAYLOAD]['email'])
        task = create_user.apply(args=[session[constants.JWT_PAYLOAD]['email']])
        logging.info(task.task_id)
        # while task.ready() == False:
        #     continue
        # done = ray.get(task)
        user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first()
        os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], str(user.id)))

    user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first() 
    logging.info(user)
    return render_template('dashboard.html',
                           userinfo=session[constants.JWT_PAYLOAD],
                           userinfo_pretty=json.dumps(session[constants.JWT_PAYLOAD], indent=4))

@app.route('/uploadsection')
@requires_auth
def uploadsection():
    user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first()
    logging.info(user)
    if user is None:
        task = create_user.apply(args=[session[constants.JWT_PAYLOAD]['email']])
        logging.info(task.task_id)
        user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first()
        os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], str(user.id)))

    user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first() 
    logging.info(user)
    return render_template('upload.html',
                           userinfo=session[constants.JWT_PAYLOAD],
                           userinfo_pretty=json.dumps(session[constants.JWT_PAYLOAD], indent=4))

@app.route('/allvideos')
@requires_auth
def allvideos():
    user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first()
    logging.info(user)
    if user is None:
        task = create_user.apply(args=[session[constants.JWT_PAYLOAD]['email']])
        logging.info(task.task_id)
        user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first()
        os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], str(user.id)))

    user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first() 
    logging.info(user)

    #get all videos for user with this user id
    return render_template('allvideos.html',
                            #allVideos=----
                           userinfo=session[constants.JWT_PAYLOAD],
                           userinfo_pretty=json.dumps(session[constants.JWT_PAYLOAD], indent=4))

@app.route('/dashboard', methods=['POST'])
@requires_auth
def upload_file():
    if request.method == 'POST':
        user = User.query.filter_by(email=session[constants.JWT_PAYLOAD]['email']).first()

        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            task = create_vid.apply(args=[user.id, filename])
            
            vid = Video.query.filter_by(user_id=user.id).all()[-1]
            # done = ray.get(task)

            task = add_vid_path.apply(args=[vid.id])

            vid = Video.query.filter_by(user_id=user.id).all()[-1]
            # done = ray.get(task)
            logging.info(vid.user_id)
            logging.info(vid.id)
            logging.info(vid.video_path)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], str(user.id), str(vid.id)))
            return render_template('allvideos.html', 
                                    prettyName=file.filename,
                                    fileName=os.path.join(str(user.id), str(vid.id)),
                                    userinfo=session[constants.JWT_PAYLOAD],
                                    userinfo_pretty=json.dumps(session[constants.JWT_PAYLOAD], indent=4))
        else:
            flash('Allowed file type is mp4')
            return redirect(request.url)
