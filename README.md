# Yolo-v3 with TensorFlow 2.0
Credit: https://github.com/penny4860/tf2-eager-yolo3


# Setup
For this installation we assume that python3 and pip is already installed and 
(if you want to use a GPU) all necessary nvidia-drivers and CUDNN. See also 
https://www.tensorflow.org/install/gpu for GPU support.

1. Create virtual environment: ```python3 -m venv env```
2. Activate venv: ```source env/bin/activate```
3. Update your pip installation: ```pip3 install --upgrade pip```
4. Install all requirements. Use requirements-gpu if a gpu is available, requirements-cpu otherwise: ```pip3 install -r requirements-cpu.txt```


# Train
To train Yolo with your own dataset, simply create a configs/*.json file which 
describes all classes, the path to your images and annotations (voc format) a 
pretrained weights file etc.

An example (configs/iis.json) is provided together with a script (scripts/downloads_iis.sh) 
to download all training and test images together with the voc annotations.

To start the training execute:
```bash
python train_eager.py -c configs/iis.json
```

To evaluate your trained model on the test set:
```bash
python eval.py -c configs/svhn.json
```


# Object dectection
To detect single objects simply call 
```bash
python pred.py -c configs/iis.json -i imgs/housenumber.jpg
```


# Video stream
```bash
python video.py -c configs/iis.json
```


# References
[1] Originial source - https://github.com/penny4860/tf2-eager-yolo3 <br />
[2] YoloV3 Paper - https://pjreddie.com/media/files/papers/YOLOv3.pdf <br />
