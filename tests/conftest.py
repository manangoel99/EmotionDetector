import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from flask import url_for

from server import app, db as db_
import pytest
from models import User, Video

@pytest.fixture
def app_def():
    app_ = app
    app_.config.from_object("config.Config")
    app_.config.update({
        'DEBUG' : False,
        'TESTING' : True,
        'SERVER_NAME' : None

    })
    app_.config['SERVER_NAME'] = "flask.dev:5000"
    ctx = app_.app_context()
    ctx.push()

    yield app_

    ctx.pop()

@pytest.yield_fixture(scope='function')
def client(app_def):
    yield app_def.test_client()

@pytest.fixture(scope='function')
def db(app_def):
    db_.drop_all()
    db_.create_all()

    db_.session.commit()

    return db_

@pytest.yield_fixture(scope='function')
def session(db):
    db.session.begin_nested()
    yield db.session
    db.session.rollback()
    db.session.close()