# CS838_Project
Semi Supervised Learning based Image Segmentation for melanoma skin cancer dataset

- cd code
- For Inception + Unet : python ./main.py ../data --mode=supTrain --epochs=90 --lr=0.00001 --weight-decay=0 --batch-size=2 
- first few filters of pretrained inception model is used as feature extractor before feeding into Unet for segmentation
- model files can be downloaded from ...
