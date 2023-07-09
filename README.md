<p align="center">
    <img src=https://github.com/dhaneshragu/i-SLR/assets/95169037/f2400d46-b302-450e-80a1-6abb6ea857cb width=200px>
</p>

I-SLR is a Machine learning powered webapp aimed at recognizing Indian and American Sign Language signs, to help the normal people who are learning sign language to test their skills.

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

## 🦾Pipeline
<p align="center">
<img alt="Training and Inference Flowchart" src="https://github.com/dhaneshragu/i-SLR/assets/106437020/c451696b-b55d-4344-8975-a4801b3688de" width=500px>
</p>
<li> Trained a custom Transformer with ASL Dataset containing 90k+ landmark data of 250 American Language signs for 40 Epochs with early stopping. (Best Epoch was 38.) </li>
<li> Used transfer learning to fine-tune the custom transformer with INCLUDE dataset containing 64 Indian Sign Language signs by just changing the last logit layer for 100 Epochs. </li>
<li>The Top 5 signs with the greatest probability are displayed.</li>

## 📈 Results
To be updated

## How to Use
### To fine-tune on a dataset

## Contributors
- [Dhanesh](https://github.com/dhaneshragu), CSE , IIT Guwahati.
- [Prabhanjan Jadhav](https://github.com/prabhanjan-jadhav), ECE , IIT Guwahati.

