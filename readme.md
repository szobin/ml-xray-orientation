# X-ray image type recognition by the `tensorflow` package

## description
for classification was used simple `tflearn` sample from this origin:
```
https://github.com/tflearn/tflearn/blob/master/examples/images/dnn.py
```
that was modified a little.

It's a sample of DNN (Deep Neural Network) that was implemented for image recognition.
Modifications:
  - implementation as python class `Classifier`
  - ImagePreprocessing was added (it's needs for normalize image)
  - ImageAugmentation was added (it's needs for avoid off small rotation and flipping influence )
To get optimal performance original images was resized to icons (32*32). It measure helps to desrease training 
time upto 1-2 minutes     

## preparing
```
git clone https://github.com/szobin/ml-xray-orientation
pip3 install -r requirements.txt
```

#### place sample images to `images` subfolder like:
/images/val/f/R12028047-a00001.png  
/images/val/f/R12004301-a00001.png
...
/images/val/s/R12064061-a00001.png
...

## usage

```
python3 classifier.py [--img path] [--epoch NN]
   path - images filename for single classification of folder for mass classification
   epoch - numver of network traning cycles   
```

## result

```
prediction result:
=====================
/images/val/f/R12028047-a00001.png - class: frontal view [0.7671600580215454 / 0.23284000158309937]
/images/val/f/R12004301-a00001.png - class: frontal view [0.7137991786003113 / 0.2862008213996887]
...
/images/val/s/R12064061-a00001.png - class: sagittal view [0.7241144180297852 / 0.27588558197021484]
/images/val/s/R12023840-a00001.png - class: sagittal view [0.6513883471488953 / 0.34861159324645996]
...
/images/val/s/R12016196-a00001.png - class: sagittal view [0.7231218814849854 / 0.27687808871269226]
=====================
```