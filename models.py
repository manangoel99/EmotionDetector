from server import db
from sqlalchemy.orm import backref

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(128), unique=True, nullable=False)
    active = db.Column(db.Boolean(), default=True, nullable=False)

    # children = db.relationship

    def __init__(self, email):
        self.email = email

class Video(db.Model):
    __tablename__ = "videos"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey(User.id, ondelete='CASCADE'), nullable=False)
    video_path = db.Column(db.String(128))
    video_title = db.Column(db.String(128))
    processed = db.Column(db.Boolean, default=False)

    # user = db.relationship(User, foreign_keys=[user_id], backref=backref("users", cascade="all,delete"))

    def __init__(self, user_id, vid_title):
        self.user_id = user_id
        self.video_title = vid_title