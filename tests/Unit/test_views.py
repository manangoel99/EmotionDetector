from flask import url_for
from models import User, Video
import constants
import shutil
import os
import io
import time
from werkzeug.datastructures import FileStorage

class TestPage(object):
    def test_home_page(self, client):
        response = client.get(url_for('home'))
        assert response.status_code == 200

    def test_user_creation_and_deletion(self, db):
        new_user = User("test_user@gmail.com")

        db.session.add(new_user)
        db.session.commit()

        assert db.session.query(User).one()
        
        user = User.query.filter_by(email="test_user@gmail.com").first()
        print(os.listdir("/home/app/public"))
        db.session.delete(user)
        db.session.commit()
        user = User.query.filter_by(email="test_user@gmail.com").first()

        assert user == None

    def test_video_creation_and_deletion(self, db):
        new_user = User("test_user@gmail.com")
        db.session.add(new_user)
        db.session.commit()

        user = User.query.filter_by(email="test_user@gmail.com").first()
        new_vid = Video(user.id, "Title")

        db.session.add(new_vid)
        db.session.commit()

        assert db.session.query(Video).one()
        id = user.id
        db.session.delete(user)
        db.session.commit()
        user = User.query.filter_by(email="test_user@gmail.com").first()

        assert user == None

        videos = Video.query.filter_by(user_id=id).all()

        assert len(videos) == 0

    def test_dashboard(self, client, db):
        with client.session_transaction() as sess:
            sess[constants.JWT_PAYLOAD] = {
                "email" : "test_user@gmail.com"
            }
            sess[constants.PROFILE_KEY] = {
                "user_id" : "test_user",
                "name" : "Test User"
            }

        res = client.get(url_for('dashboard'))
        assert res.status_code == 200
        assert "Welcome" in str(res.data)
        user = User.query.filter_by(email="test_user@gmail.com").first()
        assert user != None
        shutil.rmtree(os.path.join("/home/app/public", str(user.id)))

        db.session.delete(user)
        db.session.commit()
        user = User.query.filter_by(email="test_user@gmail.com").first()
        assert user == None

    def test_video_upload(self, client, db):
        # new_user = User(email="test_user@gmail.com")
        # db.session.add(new_user)
        # db.session.commit()
        with client.session_transaction() as sess:
            sess[constants.JWT_PAYLOAD] = {
                "email" : "test_user@gmail.com"
            }
            sess[constants.PROFILE_KEY] = {
                "user_id" : "test_user",
                "name" : "Test User"
            }

        res = client.get(url_for('dashboard'))
        assert res.status_code == 200
        
        data = {}
        data['file'] = (io.BytesIO(b"abcdef"), "./testvdo.mp4")

        response = client.post(
            url_for('upload_file'),
            data=data,
            follow_redirects=True,
            content_type='multipart/form-data'
        )

        vid = Video.query.all()[-1]

        user = User.query.filter_by(email="test_user@gmail.com").first()

        assert vid.user_id == user.id
        assert os.path.exists(os.path.join("/home/app/public", str(user.id), str(vid.id)))
        assert vid.video_path == os.path.join(str(user.id), str(vid.id))
        try:
            shutil.rmtree(os.path.join("/home/app/public", str(user.id)))
        except:
            print("Folder Not Created")
        db.session.delete(user)
        db.session.commit()


    def test_video_processing(self, client, db):
        with client.session_transaction() as sess:
            sess[constants.JWT_PAYLOAD] = {
                "email" : "test_user@gmail.com"
            }
            sess[constants.PROFILE_KEY] = {
                "user_id" : "test_user",
                "name" : "Test User"
            }

        res = client.get(url_for('dashboard'))
        assert res.status_code == 200
        my_file = FileStorage(
            stream=open(os.path.join("/home/app/tests", "testvdo.mp4")),
            filename="my_video.mp4",
            content_type="video/mpeg"
        )
        data = {}
        data['file'] = io.open("/home/app/tests/testvdo.mp4", "rb", buffering=0)
        print(my_file)
        response = client.post(
            url_for('upload_file'),
            data=data,
            follow_redirects=True,
            content_type='multipart/form-data'
        )

        vid = Video.query.all()[-1]

        user = User.query.filter_by(email="test_user@gmail.com").first()

        assert vid.user_id == user.id
        assert vid.video_path == os.path.join(str(user.id), str(vid.id))

        time.sleep(45)
        

        assert os.path.exists(os.path.join("/home/app/public", str(user.id), str(vid.id) + "_emotion.webm"))

        try:
            shutil.rmtree(os.path.join("/home/app/public", str(user.id)))
        except:
            pass
        db.session.delete(user)
        db.session.commit()