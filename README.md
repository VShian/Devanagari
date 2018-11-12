# Devanagari Optical Character Recognition
#### Dependancies:
* TensorFlow==1.5
* OpenCV
* Numpy
* Matplotlib

#### DATA:
The dataset of Devanagari characters consists of 2000 images for each of the 36 consonants (क  to ज्ञ) and numbers (0 to 9). It is available on kaggle : https://www.kaggle.com/rishianand/devanagari-character-set


_nn.py_ : is a python code to implement Neural Network

_ip.py_ : Contains all image processing techniques that are required, in the end, to extract characters from image.

_main.py_ : ocr implementation

**Specifications of NN model**: 1 input layer(1024 neurons,'ReLU'), 5 hidden layers(600,800,1000,800,600 neurons, 'ReLU') and 1 output layer(36 neurons, 'softmax')

```python
ip.get_lines(img)
```

The input is an image file in which the rows with most number of white pixels is calculated in order to separate the sentences.

```python
ip.get_chars(start, end, img)
```

Once we get lines of text, the characters are separated by calculating number of pixels vertically.

The character is resized into 32X32 pixel image and inverted. The 32X32 image is flattened into a 1024x1 array to be fed into network.

```python
checkpoint_path = training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
```
Create a checkpoint path and directory to save the model after training and to load it during testing.

```python
nn.load_data() 
```

returns 'train' and 'test' data as numpy arrays

```python
nn.create_model()
```

returns _model_ with the mentioned specs.



We have already saved the model in a [training diectory](https://www.dropbox.com/sh/7h61l3eoli4byra/AACniJmdQ-MOe-hcksrCezoXa?dl=0). It contains checkpoint(synaptic weights) of the _model_ after 15 epochs (which is quite less.. but it can be used as initial weights to the model and can be continued from there to any number of epochs, say 50,for better results) which can be loaded, as shown below, for testing.

```python
nn.load_weights(checkpoint_path)
```

_^ we assume background and letters to be in white and black colours respectively._

_^^ to satisy our _model_ which requires background and letters to be in black and white colours respectively._
