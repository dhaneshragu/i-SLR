from flask import Flask,render_template,Response,request
from threading import Thread
from mp_funcs import *
import utils
import model
import mediapipe as mp
import cv2
import torch
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import SequentialSampler, RandomSampler
import torch.nn as nn
ROWS_PER_FRAME = 543

cap = cv2.VideoCapture(0)
streaming = False
df = pd.DataFrame()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
final_landmarks = []
app=Flask(__name__)
mp_holistic = mp.solutions.holistic


def generate_frames():
    global final_landmarks
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while streaming:
            if cap.isOpened():
                success, frame = cap.read()

                if not success:
                    break
            
                image, results = mediapipe_detection(frame, holistic)
                draw(image, results)
                landmarks = extract_coordinates(results)
                final_landmarks.extend(landmarks)
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
net = model.Net()
net.to(device)
net.load_state_dict(torch.load('00000038.model.pth', map_location=torch.device('cpu'))['state_dict'])

net_plus = model.Net(num_class=250)
net_plus.logit = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 64)
)
net_plus.to(device)
net_plus.load_state_dict(torch.load('ft_00000100.model.pth', map_location= torch.device('cpu'))['state_dict'])

def inf_null_collate(batch):
    batch_size = len(batch)
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]
    return d
    

@app.route('/ISL')
def ISL():
    return render_template('ISL.html',lang_name="Indian Sign Language")

@app.route('/ASL')
def ASL():
    return render_template('ISL.html',lang_name="American Sign Language")

@app.route('/')
def index():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    global streaming
    if streaming:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(send_image(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recordingISL',methods=['POST','GET'])
def recordingISL():
    global streaming, final_landmarks
    if request.method == 'POST':
        if request.form.get('submit-1') == 'Start':
            if not streaming:
                streaming = True
                cap.open(0)
        if request.form.get('submit-2') == 'Stop':
            streaming = False
            cap.release()
            cv2.destroyAllWindows()
            data = pd.DataFrame(final_landmarks, columns=['x','y','z'])

            n_frames = int(len(data) / ROWS_PER_FRAME)
            xyz = data.values.reshape(n_frames, ROWS_PER_FRAME, 3).astype(np.float32)
            xyz = xyz - xyz[~np.isnan(xyz)].mean(0,keepdims=True) 
            xyz = xyz / xyz[~np.isnan(xyz)].std(0, keepdims=True)
            xyz = torch.from_numpy(xyz).float()
            xyz = utils.pre_process(xyz)
            r = {}
            r['index'] = 0
            r['xyz'  ] = xyz
            valid_loader = DataLoader(
                [r],
                sampler = SequentialSampler([r]),
                batch_size  = 1,
                drop_last   = False,
                num_workers = 0,
                pin_memory  = False,
                collate_fn = inf_null_collate,
            )
            net_plus.eval()
            preds = []
            for t, batch in enumerate(valid_loader):
                net_plus.output_type = ['inference']
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled = True):
                        output = net_plus(batch)
                        top_values, top_indices = torch.topk(output['sign'].detach().cpu(), k=5)
                        preds = top_indices
            final_landmarks=[]
            f = open('isl_label2sign.json')
            data = json.load(f)
            signs = []
            for l in preds[0]:
                signs.append(data[str(l.item())])
            return render_template('ISL.html',lang_name ="Indian Sign Language" , preds=signs) # Change to the top5 predictions list 

    elif request.method == 'GET':
        return render_template('ISL.html',lang_name ="Indian Sign Language")

    return render_template('ISL.html',lang_name ="Indian Sign Language")


@app.route('/recordingASL',methods=['POST','GET'])
def recordingASL():
    global streaming, final_landmarks
    if request.method == 'POST':
        if request.form.get('submit-1') == 'Start':
            if not streaming:
                streaming = True
                cap.open(0)
        if request.form.get('submit-2') == 'Stop':
            streaming = False
            cap.release()
            cv2.destroyAllWindows()
            data = pd.DataFrame(final_landmarks, columns=['x','y','z'])
            final_landmarks = []
            n_frames = int(len(data) / ROWS_PER_FRAME)
            xyz = data.values.reshape(n_frames, ROWS_PER_FRAME, 3).astype(np.float32)
            xyz = xyz - xyz[~np.isnan(xyz)].mean(0,keepdims=True) 
            xyz = xyz / xyz[~np.isnan(xyz)].std(0, keepdims=True)
            xyz = torch.from_numpy(xyz).float()
            xyz = utils.pre_process(xyz)
            r = {}
            r['index'] = 0
            r['xyz'  ] = xyz
            valid_loader = DataLoader(
                [r],
                sampler = SequentialSampler([r]),
                batch_size  = 1,
                drop_last   = False,
                num_workers = 0,
                pin_memory  = False,
                collate_fn = inf_null_collate,
            )
            net.eval()
            preds = []
            for t, batch in enumerate(valid_loader):
                net.output_type = ['inference']
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled = True):
                        output = net(batch)
                        top_values, top_indices = torch.topk(output['sign'].detach().cpu(), k=5)
                        preds = top_indices

            f = open('asl_label2sign.json')
            data = json.load(f)
            signs = []
            for l in preds[0]:
                signs.append(data[str(l.item())])
            return render_template('ISL.html',lang_name ="American Sign Language",preds=signs) # Change to top5 preds list
    elif request.method == 'GET':
        return render_template('ISL.html',lang_name ="American Sign Language")

    return render_template('ISL.html',lang_name ="American Sign Language")


if __name__ == '__main__':
    app.run(debug=True)
