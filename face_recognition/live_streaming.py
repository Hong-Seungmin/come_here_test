# live_streaming.py
from datetime import timedelta

from flask import Flask, render_template, Response, request, redirect, session, url_for, jsonify
import face_recog
import threading
import face_recog_classifier
import os
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, db
from User import User

cred = credentials.Certificate("firebase/comehere-bb95d-firebase-adminsdk-5k4ey-609741948b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://comehere-bb95d-default-rtdb.firebaseio.com/'
})

app = Flask(__name__)
app.secret_key = 'zzzaaeeee'
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=5)
faceClassifier = face_recog_classifier.FaceClassifier('http://192.168.101.21:3080/?action=stream', 0.45, 0.25)


@app.route('/')
def home():
    """ Session control"""
    if not session.get('username'):
        return render_template('home.html')
    else:
        username = session['username']
        nickname = session['nickname']
        return render_template('home.html', username=username, nickname=nickname)


@app.route('/stream_video')
def stream_video():
    return render_template('stream_video.html')


@app.route('/upload_img', methods=['POST'])
def upload_img():
    file = request.files['uploaded_file']
    nickname = request.form.get('nickname')

    print('닉네임: ', nickname)

    filename = secure_filename(file.filename)
    os.makedirs('./member', exist_ok=True)
    os.makedirs('./member/' + nickname, exist_ok=True)
    file.save(os.path.join('./member/' + nickname, filename))

    return 'OK'


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login Form"""
    if request.method == 'GET':
        return render_template('home.html')
    else:
        username = request.form['username']
        password = request.form['password']

        ref = db.reference('server/data/users')
        user = ref.order_by_key().equal_to(username).get()

        if len(user) == 0:
            return jsonify({'result': 'no'}), 200

        print('user : ', username)
        if user[username]['password'] == password:

            session['username'] = username
            session['nickname'] = user[username]['nickname']

            return jsonify({'result': 'yes', 'username': username, 'nickname': user[username]['nickname']}), 200
        else:
            return jsonify({'result': 'no'}), 200


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register Form"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        nickname = request.form['nickname']

        ref = db.reference('server/data/users')
        ref.child(username).set({
            'password': password,
            'nickname': nickname
        })

        session['username'] = username
        session['nickname'] = nickname

    #     return redirect(url_for('home'))
    # return render_template('register.html')
    return '', 204


@app.route("/logout", methods=['GET', 'POST'])
def logout():
    """Logout Form"""
    if session['username']:
        session.pop('username', None)
    return redirect(url_for('home'))


def gen():
    while True:
        jpg_bytes = faceClassifier.jpg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')


if __name__ == '__main__':
    t = threading.Thread(target=faceClassifier.start_recog)
    t.start()
    app.run(host='0.0.0.0')
    faceClassifier.running = False
