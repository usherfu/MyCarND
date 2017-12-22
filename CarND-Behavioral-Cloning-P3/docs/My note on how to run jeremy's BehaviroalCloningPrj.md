My note on how to run [jeremy's CarND-Behaviroal-Cloning-Project](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project)

PS: Also check out following Behaviroal Cloning projects:
 - [upul](https://github.com/upul/Behavioral-Cloning)
 - [PaulHeraty](https://github.com/PaulHeraty)
 - [!!!use ResNet and has another project for RC car](https://github.com/datlife/behavioral-cloning)
 - [!!!Teaching a Machine to Steer a Car](https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c)    
# Problem Encoutered

#### Cannot read Keras model 
Uninstall Keras 2.0 in the docker image and reinstall Keras 1.2.0, as the model is saved in Keras 1.2 version. and it seems not compatible with Keras 2.x

```bash
  pip3 uninstall -y keras
  pip3 install keras==1.2.0
```

#### Install opencv and opencv_contrib. Also enable cv2.imshow()
Following steps try to install opencv and enable run of cv2.imshow. Following steps should be executed in sequence or part of dockerfile. They are currently implemented in [ml-docker](https://github.com/usherfu/ml-docker)

```bash
apt-get update && apt-get dist-upgrade && apt-get autoremove
apt-get install -y libqt4-dev build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libtbb-dev

apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
python2.7-dev python3.5-dev

ARG OPENCV_VERSION 3.3.1
#download opencv and opencv_contrib
wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip  && unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && unzip opencv_contrib.zip

cd ~/opencv-${OPENCV_VERSION}
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON -D CUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-${OPENCV_VERSION}/modules -D PYTHON_EXECUTABLE=/usr/bin/python3 ..


make -j"$(nproc)"  && \
make install && \
ldconfig && \

mv /usr/local/lib/python3.5/dist-packages/cv2.cpython-35m-x86_64-linux-gnu.so /usr/local/lib/python3.5/dist-packages/cv2.so
cd /usr/local/lib/python3.5 && mkdir site-packages 
ln -s /usr/local/lib/python3.5/dist-packages/cv2.so /usr/local/lib/python3.5/site-packages/cv2.so

```


#### How to use virtualenv with opencv.

```bash
cd ~
apt-get install -y python-pip && pip install --upgrade pip
pip install virtualenv virtualenvwrapper
rm -rf ~/.cache/pip
#edit .bashrc file
source ~/.bashrc
mkvirtualenv cv -p python3
pip install numpy
ln -s /usr/local/lib/python3.5/site-packages/cv2.so ~/.virtualenvs/cv/lib/python3.5/site-packages/cv2.so
```
check details [here, step4](https://medium.com/@debugvn/installing-opencv-3-3-0-on-ubuntu-16-04-lts-7db376f93961)


#### To test cv2.imshow, create a python file with following code

```python
import cv2
img = cv2.imread('/sharedfolder/images/model_diagram.jpeg')
cv2.imshow('Window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### To test matplotlib.pyplot.imshow, create a python file with following code

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('/sharedfolder/images/nVidia_model.png')
imgplot = plt.imshow(img)
plt.show()
```

# Reference:
- Add PYTHON_EXECUTABLE  for cmake command
	https://stackoverflow.com/questions/36201282/install-opencv-for-python3

- **How to install OpenCV and OpenCV_Contrib from source code**
	https://medium.com/@debugvn/installing-opencv-3-3-0-on-ubuntu-16-04-lts-7db376f93961

- Add CUDA_CUDA_LIBRARY for cmake command
	https://github.com/opencv/opencv/issues/6577

- How to configure DISPLAY for docker container
	https://paddy-hack.gitlab.io/posts/running-dockerized-gui-applications/

- Showing pictures with Matplotlib
	https://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/
	
- Set env variable "export QT_X11_NO_MITSHM=1" to avoid blank screen (no picture showing)
	https://github.com/P0cL4bs/WiFi-Pumpkin/issues/53
	
- [CarND Slack Channel](https://carnd.slack.com/messages/C2HQUQN80/)