import time

import cv2
import dlib
import face_recognition
import numpy as np
import ray
from imutils import face_utils
from keras.models import load_model
import glob
from preprocess import preprocess_input
from utils import *

ray.init(num_cpus=8, num_gpus=1, ignore_reinit_error=True, webui_host='0.0.0.0')
time.sleep(2)

@ray.remote
class Model(object):
	from keras.models import load_model

	def __init__(self):
		emotion_model_path = './model.hdf5'
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
		return each_face_emotion

	# def getVideo(self, cap):
	#     all_emotions = []
	#     while cap.isOpened():
	#         ret, frame = cap.read()
	#         if frame is None:
	#             break
	#         all_emotions.append(self.predictFrame(frame))
	#     return all_emotions
def process_vid(vid_path):
	cap = cv2.VideoCapture(vid_path)
	all_emotions = []
	start = time.time()
	detect = Model.remote()
	while cap.isOpened():
		ret, frame = cap.read()
		if frame is None:
			break
		all_emotions.append(detect.predictFrame.remote(frame))

	return all_emotions

img_map,height,widht,layers = {},-1,-1,-1
def generate_emoji_video(emotion_text):
	out = cv2.VideoWriter('emoji_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
	for emotion in emotion_text:
		for it in emotion:
			if it == None:
				break
			out.write(img_map[it])
	out.release()

if __name__ == '__main__':
	for file in glob.glob('../emoji/*.png'):
		img = cv2.imread(file)
		height, width, layers = img.shape
		size = (width,height)
		img_map[(file.split('/')[2]).split('.')[0]] = img
	start = time.time()
	emotions = process_vid("./testvdo.mp4")
	end = time.time()
	print("==================")
	print(end - start)
	print(len(emotions))
	print("==================")
	# get_emoji_video(emotions)
	generate_emoji_video(ray.get(emotions))
	# print(ray.get(all_emotions))