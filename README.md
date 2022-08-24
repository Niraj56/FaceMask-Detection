# FaceMask Detection
<img src="https://i.imgur.com/2nc0dHJ.gif" />

I've built `FaceMask Detection` system with `OpenCV`, `TensorFlow`. I didn’t use any ‘already trained models’ as I wanted to make it accurate and very simple. I used my own model with my own dataset in order to detect face masks statically and on real time streams.

## REQUIREMENTS:

* Python 3.7 or later
* Numpy
* OpenCV (CV2)
* Tensorflow
* PIL
* PyQt5

> **Note**
> Make sure you have installed / setup everything correctly and have correct python paths.

## Installing Libraries

1. `Numpy`

```
pip install numpy
```

2. `OpenCv`

```
pip install opencv-python
```

3. `Tensorflow`

```
pip install tensorflow
```

4. `PIL`

```
pip install pillow
```

5. `PyQt5`

```
pip install PyQt5
```

## How to Run
1. For the python script,	> ```python Face_mask_detector.py```
2. Or, simply execute the ```Face_mask_detector.exe```

> **Note**
> If the camera does not turn off even after closing program follow these steps: Enter this in the cmd > ```setx OPENCV_VIDEOIO_PRIORITY_MSMF 0```
>  and Restart the pc
