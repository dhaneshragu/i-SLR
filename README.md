<p align="center">
    <img src=https://github.com/dhaneshragu/i-SLR/assets/95169037/f2400d46-b302-450e-80a1-6abb6ea857cb width=200px>
</p>

i-SLR is a Machine learning powered webapp aimed at recognizing Indian and American Sign Language signs, to help the normal people who are learning sign language to test their skills.

This app can currently recognize 250 American Signs and 64 Indian Sign Signs.

## 📗 Tech stack
<p>
<img src="https://img.shields.io/badge/Pytorch-EE4C2C?logo=pytorch&logoColor=white&style=flat" />
<img src="https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white&style=flat" />
<img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white&style=flat" />
<img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=flat" />
<img src="https://img.shields.io/badge/OpenCV-D95D39?logo=opencv&logoColor=white&style=flat" />
<img src="https://img.shields.io/badge/Python-2596be?logo=python&logoColor=white&style=flat" />
<img src="https://img.shields.io/badge/HTML-e44d26?logo=html5&logoColor=white&style=flat" />
<img src="https://img.shields.io/badge/JavaScript-f3db1c?logo=javascript&logoColor=white&style=flat" />
<img src="https://img.shields.io/badge/CSS-264de4?logo=css3&logoColor=white&style=flat" />
<img src="https://img.shields.io/badge/Streamlit-DB423D?logo=streamlit&logoColor=white&style=flat" />
</p>

## 📽️ Video Demo
https://github.com/dhaneshragu/i-SLR/assets/95169037/442d6144-e3c1-4743-880b-353ac2c05601

## 🧪 Pipeline
<p align="center">
<img alt="Training and Inference Flowchart" src="https://github.com/dhaneshragu/i-SLR/assets/106437020/c451696b-b55d-4344-8975-a4801b3688de" width=500px>
</p>
<li> Trained a custom Transformer with ASL Dataset containing 90k+ landmark data of 250 American Language signs for 40 Epochs with early stopping. (Best Epoch was 38.) </li>
<li> Used transfer learning to fine-tune the custom transformer with INCLUDE dataset containing 64 Indian Sign Language signs by just changing the last logit layer for 100 Epochs. </li>
<li>The Top 5 signs with the greatest probability are displayed.</li>

## 📈 Results
Achieved top 5 accuracies of **97.32%** and **95.11%**, and cross-entropy losses of **0.534** and **0.766** for training and validation, respectively for the ISL model.

<img alt="ISL evaluation" src="https://github.com/dhaneshragu/i-SLR/assets/106437020/b1817376-ef0b-4af1-b407-bc1fc03bfb7b" width=800px>


## 🤩 Build your own sign language recognizer using i-SLR
- First of all `git clone` this repository and cd to the appropriate folder
- Go to `Dataset-Creation Folder` . There are 2 python scripts `dataset_creater.py` and `preprocess.py`. Run `dataset_creator.py` while having the dataset videos in the directory structure as shown :
```
/Dataset-Creation
└── INCLUDE
    ├── Sign-Category-1
    └── Sign-Category-2
        ├── 1.Sign-Name-1
        ├── 2.Sign-Name-2
            └── Sign-Video-1.mp4
            └── Sign-Video-2.mp4
        └── 3.Sign-Name-3
└── dataset_creator.py
└── preprocess.py
```
- All the videos will be preprocessed with mediapipe and landmarks will be saved in a csv called `train-preprocessed.csv`.
- Go to the fine-tuning section of the `iSLR-Notebook.ipynb` and replace the train-csv URL with the `train-preprocessed.csv` path.
- Run `make_json.py` to store signs with respect to their labels into a json file. 
- Run the notebook and you can get the `model.pth` file which can be replaced in flask webapp to generate predictions !!!
## 👤 To use the app :
- Make sure you are in the cloned repository folder. 
- In terminal , type `python app.py` and then the flask webapp will start in your browser.

  <img width="700" alt="image" src="https://github.com/dhaneshragu/i-SLR/assets/95169037/4ec936cf-24a2-4bb2-a211-e24858d8aa79">
- Navigate to **Indian Sign Language** and **American Sign Language** section, click **Start** and sign and click **Stop** when you are done.

  <img width="602" alt="image" src="https://github.com/dhaneshragu/i-SLR/assets/95169037/b212e175-a947-416f-9c8c-74360d44a7d6"> 
- Viola!! You will get the top-5 predictions of the sign you made.

  <img width="602" alt="image" src="https://github.com/dhaneshragu/i-SLR/assets/95169037/85df4b01-3b3c-4dc3-936a-dc342ace8aee">

## 🦾 Contributors
- [Dhanesh](https://github.com/dhaneshragu), CSE , IIT Guwahati.
- [Prabhanjan Jadhav](https://github.com/prabhanjan-jadhav), ECE , IIT Guwahati.

## 🌟 Stay connected
Don't forget to ⭐️ star this repository to show your support and stay connected for future updates!
