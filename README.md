# TensorFlowProjects
Some Deep Learning programs based on TensorFlow. 

Here is how the dictory should look like:

    root
    ├── Gen_pic
    ├── MNIST_data
    │   ├── t10k-images-idx3-ubyte.gz
    │   ├── t10k-labels-idx1-ubyte.gz
    │   ├── train-images-idx3-ubyte.gz
    │   └── train-labels-idx1-ubyte.gz
    ├── Model
    ├── selfmade_pic
    ├── xxx.py
    ...

  * ./selfmade_pic/ includes handwritten digits that I made by Photoshop. They are all 28*28 pixels, 8-bit gray scale PNG. 
  * ./Model/ includes saved TensorFlow models. 
  * CNN_MNSIT.py is a handwritten digit recognition CNN network trained by MNIST set. (Accuracy: 99.01%)
  * DNN_XOR.py a simple MLP network trained to do exclusive-or. 
  * GAN_dgtGen.py gennerates digits based on MNIST. https://www.bilibili.com/video/av23221006
  * LinearRegression.py calculates regressed function and charts the plot. 
  * RNN_SeqRec.py is RNN for certain number sequence identification. 
  * mnist2png.py convert MNIST data to .png picture. In addition, generating Gaussian image noise. 
  
