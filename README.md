# Captcha Solver
This Captcha Solver is made using the deep learning and computer vision libraries of python.The model has been trained on Google Colab using the handwritten characters dataset provided by [emnist](https://www.kaggle.com/crawford/emnist). The dataset has been modified to remove some similar looking characters like O and 0. The model has been able to acheive a validation accuracy of upto 99% and a test accuracy of 92%.
## Character Set
The character set on which the model has been trained is available in the `characters.txt` file.
## Features of the model
The special features of the model includes:
1. Decoding captchas with lots of noise in the form of lines and circular dots
2. Decoding captchas with characters rotated by 45 deegrees
3. Decoding a diverse range of captchas ranging from very simple ones having only 2 or 3 characters to complex ones having characters of variable thichness and size

A few sample captchas has been provided in the Sample_captchas folder.
All the python libraries required to run the model have been included in the `requirements.txt` file.You can directly run this file on your terminal to get them installed.
To test the model you can simply clone this repo and add your image paths in the `decode_captcha.py` file and run that file.
