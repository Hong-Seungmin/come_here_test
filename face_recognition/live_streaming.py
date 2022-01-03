# live_streaming.py

from flask import Flask, render_template, Response
import face_recog
import threading
import face_recog_classifier

app = Flask(__name__)
faceClassifier = face_recog_classifier.FaceClassifier('http://192.168.101.21:3080/?action=stream', 0.4, 0.25)

print('11111111111111111111111111111111111111111')

@app.route('/')
def index():
    return render_template('index.html')


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
    print('22222222222222222222222')
    t = threading.Thread(target=faceClassifier.start_recog)
    t.start()
    print('zxczxc')
    app.run(host='0.0.0.0', debug=True)
    print("endendendend")
    faceClassifier.running = False
    print('zzzzzzzzzzzzzzzz')
