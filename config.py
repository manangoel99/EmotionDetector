import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite://")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = "./public"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    CELERY_BROKER_URL = 'redis://redis:6379/0'
    CELERY_BROKER_BACKEND = 'redis://redis:6379/0'
