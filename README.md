# Self-supervised Representation Learning for Lesion Segmentation 
Self Supervised Learning based Image Segmentation for melanoma skin cancer dataset

- cd code
- For Self Supervised Learning phase : python ./main.py ../data --mode=preTrain --epochs=90 --lr=0.00001 --weight-decay=0
- For Fully Supervised Learning phase : python ./main.py ../data --mode=supTrain --epochs=90 --lr=0.00001 --weight-decay=0 --batch-size=2 
