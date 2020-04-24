import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite://")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = "./public"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    CELERY_BROKER_URL = 'redis://redis:6379/0'
    CELERY_BROKER_BACKEND = 'redis://redis:6379/0'
    CELERY_PREFETCH_MULTIPLIER = 1
    CELERY_MAX_TASKS_PER_CHILD = 10
    CELERY_MAX_MEMORY_PER_CHILD = 150000
    SERVER_NAME = None
