# The Emotion Detector App

This is a web app built as a part of our Software Engineering course which provides an interface to users to upload videos and see the emotions on each face in the video. The web interface is created using [Flask](https://flask.palletsprojects.com/en/1.1.x/), the database is mantained using [PostgreSQL](https://www.postgresql.org/) and task management is done using [Celery](http://www.celeryproject.org/) which uses is [Redis](https://redis.io/) as a message broker. All transactions with the database are done using a celery task queue.
<br>
The processing of videos is done asynchronously after video upload to make sure there is no lag in user experience. The processing of the video is done framewise and the frames are distributed across CPUs to make the processing as fast as possible. The parallel processing of frames gave a 10x speed up in the processing time. The parallel processing was achieved using [Ray](https://ray.io/) which was built specifically for distributed processing especially in Machine Learning.
<br>
For the emotion detection, we divided the task into 2 parts
- Face Detection : The face detection in each frame is done by the [face-recognition](https://github.com/ageitgey/face_recognition) library which uses [dlib](http://dlib.net/)'s state of the art face detection model which has 99.38% accuracy. It returns the coordinates of the faces in each frame and then we run our emotion detection model on these faces.
- Emotion Detection : For emotion detection we obtained our dataset from a [Kaggle Challenge](https://www.kaggle.com/c/emotion-detection-from-facial-expressions) which gave us a clean, labelled dataset. The model was built in keras with a tensorflow backend and it uses seperable convolution to get the image features on which linear layers are applied followed by a softmax to get probabilities of each emotion. We deployed the model using keras and it takes each frame as input and gives the emotions on each face in the frame as output.
- We also stored how each emotion varies during the duration of the video which can be used to draw conclusions about the reactions of people in the video. For example : A video of students in a class may give information about how the students interest level changes through the duration of the class.
----------------------

The Horizonatal expansion of the system can be done quite easily.
- Celery : Changing the number of processes and memory in ```docker-compose.yml``` for the celery worker.
- Ray : In the ```ray.init()``` command in ```server.py``` we can easily change the number of CPUs and the RAM available.

The vertical expansion can be done by :
- Celery : Creating more celery workers in ```docker-compose.yml``` with the address of each celery worker on the master node.
- Ray : 
    - Set up a ray worker on each of the other nodes in the cluster.
    - In ```ray.init()``` on the master node, specify the address of each ray worker.
    - More details [here](https://ray.readthedocs.io/en/latest/using-ray-on-a-cluster.html)
# Running the App

Install Docker from https://docs.docker.com/engine/install/ubuntu/
<br>
Install docker-compose from https://docs.docker.com/compose/install/\
<br>
To run docker without sudo use https://docs.docker.com/engine/install/linux-postinstall/
<br>
Once installed 
```
git clone https://github.com/manangoel99/EmotionDetector.git
cd EmotionDetector
docker-compose up --build -d
docker-compose exec web python manage.py create_db
```
The last command sets up an empty database

-------------------------------

# Devlopment

Set up the project as mentioned in the previous section. Edit the files as required and the server will restart with the new changes.
<br>

To run commands inside the docker-container
```
docker-compose exec web bash
```
To check database
```
docker-compose exec db psql --username=hello_flask --dbname=hello_flask_dev
```
```\dt``` shows list of tables. Use simple sql commands for query.
<br>
Please do not push the public folder.
<br>
To run tests
```
docker-compose up --build -d
docker-compose exec web py.test
```
