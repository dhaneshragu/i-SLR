from flask import Flask,render_template,Response,request
from threading import Thread
import mp_funcs
import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
streaming = False

app=Flask(__name__)
mp_holistic = mp.solutions.holistic


def generate_frames():
 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while streaming:
            if cap.isOpened():
                success, frame = cap.read()

                if not success:
                    break
            
                image, results = mp_funcs.mediapipe_detection(frame, holistic)
                mp_funcs.draw(image, results)
                #landmarks = mp_funcs.extract_coordinates(results)
                #print(len(landmarks))
                #final_landmarks.extend(landmarks)
                #cv2.imshow('Hello',image)
                #cv2.imshow('Webcam Feed',image)
                try:
                    _, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
                except Exception as e:
                    pass


def send_image():
    img_path = "no_cam.png"  
    img = cv2.imread(img_path)

    # Convert the image to JPEG format
    _, buffer = cv2.imencode('.jpg', img)
    image_data = buffer.tobytes()

    yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image_data + b'\r\n')


@app.route('/ISL')
def ISL():
    return render_template('ISL.html',lang_name="Indian Sign Language")

@app.route('/ASL')
def ASL():
    return render_template('ISL.html',lang_name="American Sign Language")

@app.route('/')
def index():
    return render_template('blog-post.html')

@app.route('/video_feed')
def video_feed():
    global streaming
    if streaming:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(send_image(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recordingISL',methods=['POST','GET'])
def recordingISL():
    global streaming
    if request.method == 'POST':
        if request.form.get('submit-1') == 'Start':
            if not streaming:
                streaming = True
                cap.open(0)
        if request.form.get('submit-2') == 'Stop':
            streaming = False
            cap.release()
            cv2.destroyAllWindows()
            return render_template('ISL.html',lang_name ="Indian Sign Language" , preds=['hello']) # Change to the top5 predictions list 

    elif request.method == 'GET':
        return render_template('ISL.html',lang_name ="Indian Sign Language")

    return render_template('ISL.html',lang_name ="Indian Sign Language")


@app.route('/recordingASL',methods=['POST','GET'])
def recordingASL():
    global streaming
    if request.method == 'POST':
        if request.form.get('submit-1') == 'Start':
            if not streaming:
                streaming = True
                cap.open(0)
        if request.form.get('submit-2') == 'Stop':
            streaming = False
            cap.release()
            cv2.destroyAllWindows()
            return render_template('ISL.html',lang_name ="American Sign Language",preds=['bye']) # Change to top5 preds list
    elif request.method == 'GET':
        return render_template('ISL.html',lang_name ="American Sign Language")

    return render_template('ISL.html',lang_name ="American Sign Language")


if __name__ == '__main__':
    app.run(debug=True)