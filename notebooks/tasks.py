from celery.task import task
import cv2
from model import Model
from celery import group


@task
def process_vid(vid_path):
	cap = cv2.VideoCapture(vid_path)
	all_emotions = []
	detect = Model()
	frames = []
	while cap.isOpened():
		ret, frame = cap.read()
		frames.append(frame)

	g = group([Model.predictFrame.s(detect, frame) for frame in frames])
	result = g.apply_async()

	while result.ready() is False:
		continue
	
	print(result.get())
	return all_emotions