# Keras Rubik's cube detection and sides recognition

This is a prototype of keras model that detect and recognize Rubik's cube based on the small (~136k param) 
segmentation u-net like model. Cube should be rotated so camera see three sides. 

Small size allows to run this model on pure JavaScript in user browser: [https://github.com/krustnic/rubikjs](https://github.com/krustnic/rubikjs)

Demo video:

[![Demo](https://img.youtube.com/vi/s2JirlnwtwE/0.jpg)](https://www.youtube.com/watch?v=s2JirlnwtwE)

Model predict two segmentation masks:
* Cube segmentation mask
* Single top cube corner

There is some OpenCV postprocessing after which we got six hexagon points (six cube corners) and cut the sides. 

**Warning! At the moment there is no dataset publicly available and code can be unreproducible.**   