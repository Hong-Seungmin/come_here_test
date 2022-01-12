# live_streaming.py

from flask import Flask, render_template, Response, request
import face_recog
import threading
import face_recog_classifier
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
faceClassifier = face_recog_classifier.FaceClassifier('http://192.168.101.21:3080/?action=stream', 0.45, 0.25)
print('start live')

@app.route('/')
def login():
    return render_template('login.html')


@app.route('/stream_video')
def stream_video():
    return render_template('stream_video.html')


@app.route('/upload_img', methods=['POST'])
def upload_img():
    file = request.files['uploaded_file']

    filename = secure_filename(file.filename)
    os.makedirs('./member', exist_ok=True)
    os.makedirs('./member/mr2018048', exist_ok=True)
    file.save(os.path.join('./member/mr2018048', filename))

    return 'OK'


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login Form"""
    if request.method == 'GET':
        return render_template('login.html')
    else:
        name = request.form['username']
        passw = request.form['password']
        try:
            data = User.query.filter_by(username=name, password=passw).first()
            if data is not None:
                session['logged_in'] = True
                return redirect(url_for('home'))
            else:
                return 'Dont Login'
        except:
            return "Dont Login"


@app.route('/register/', methods=['GET', 'POST'])
def register():
    """Register Form"""
    if request.method == 'POST':
        new_user = User(
            username=request.form['username'],
            password=request.form['password'])
        db.session.add(new_user)
        db.session.commit()
        return render_template('login.html')
    return render_template('register.html')


@app.route("/logout")
def logout():
    """Logout Form"""
    session['logged_in'] = False
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
