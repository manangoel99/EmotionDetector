import time

import cv2
import dlib
import face_recognition
import numpy as np
import ray
from imutils import face_utils
import glob
from preprocess import preprocess_input
from utils import *

ray.init(num_cpus=16, num_gpus=1)
time.sleep(2)

@ray.remote
class Model(object):

	def __init__(self):
		from keras.models import load_model
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
	detect = Model.remote()
	while cap.isOpened():
		ret, frame = cap.read()
		if frame is None:
			break
		emotion = detect.predictFrame.remote(frame)
		all_emotions.append(emotion)

	return all_emotions

def generate_emotion_video(ray_list,file_path):
	cap = cv2.VideoCapture(file_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	while cap.isOpened():
		ret,frame = cap.read()
		if frame is None:
			break
		height, width, layers = frame.shape
		size = (width,height)
		break
	cap.release()

	out = cv2.VideoWriter('emotion_video.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
	for iterx in ray_list:
		out.write(iterx[1])

if __name__ == '__main__':
	start = time.time()
	emotions = process_vid("./testvdo.mp4")
	end = time.time()
	print("==================")
	print(end - start)
	print(len(emotions))
	print("==================")
	# get_emoji_video(emotions)
	# print(ray.get(all_emotions))