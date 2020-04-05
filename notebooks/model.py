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

ray.init(num_cpus=8, num_gpus=1, ignore_reinit_error=True)
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

frame_window = 10
emotion_offsets = (20, 40)

def generate_emotion_video(emotion_text_arr,file_path):
	cap = cv2.VideoCapture(file_path)
	while cap.isOpened():
		ret,frame = cap.read()
		if frame is None:
			break
		height, width, layers = frame.shape
		size = (width,height)
		break

	cap.release()
	c = 0
	cap = cv2.VideoCapture(file_path)
	out = cv2.VideoWriter('emotion_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
	emotion_target_size = (64,64)
	while cap.isOpened() and c<len(emotion_text_arr):
		ret, frame = cap.read()
		if frame is None:
			break
		if emotion_text_arr[c] == []:
			c+=1
			out.write(frame)
			continue
		gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		detector = dlib.get_frontal_face_detector()
		faces = detector(rgb_image)
		j = 0
		for face_coordinates in faces:
			x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
			gray_face = gray_image[y1:y2, x1:x2]
			try:
				gray_face = cv2.resize(gray_face, emotion_target_size)
			except:
				continue

			gray_face = preprocess_input(gray_face, True)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)

			if emotion_text_arr[c][j] == 'angry':
				color = np.asarray((255, 0, 0))
			elif emotion_text_arr[c][j] == 'sad':
				color = np.asarray((0, 0, 255))
			elif emotion_text_arr[c][j] == 'happy':
				color = np.asarray((255, 255, 0))
			elif emotion_text_arr[c][j] == 'surprise':
				color = np.asarray((0, 255, 255))
			elif emotion_text_arr[c][j] == 'disgusted':
				color = np.asarray((0, 255, 0))				
			else:
				color = np.asarray((255, 255, 255))

			color = color.astype(int)
			color = color.tolist()

			name = emotion_text_arr[c][j]
			
			draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
			draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, name,color, 0, -45, 0.5, 1)
			j += 1

		frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		c+=1
		out.write(frame)
	
	cap.release()

if __name__ == '__main__':
	start = time.time()
	emotions = process_vid("./testvdo.mp4")
	end = time.time()
	print("==================")
	print(end - start)
	print(len(emotions))
	print("==================")
	generate_emotion_video(ray.get(emotions),"./testvdo.mp4")
