from server import celery, db
from models import User,Video
import logging

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