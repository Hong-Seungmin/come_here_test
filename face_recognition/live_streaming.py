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
def index():
    return render_template('index.html')


@app.route('/upload_img', methods=['POST'])
def upload_img():
    file = request.files['uploaded_file']

    filename = secure_filename(file.filename)
    os.makedirs('./member', exist_ok=True)
    os.makedirs('./member/mr2018048', exist_ok=True)
    file.save(os.path.join('./member/mr2018048', filename))

    return 'OK'


def gen():
    while True:
        jpg_bytes = faceClassifier.jpg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    t = threading.Thread(target=faceClassifier.start_recog)
    t.start()
    app.run(host='0.0.0.0')
    faceClassifier.running = False
