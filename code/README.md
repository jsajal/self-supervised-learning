Used pre-trained inception network as feature extractor and U-net on top for segmentation.
run: python ./main.py ../data --mode=supTrain --epochs=90 --lr=0.00001 --weight-decay=0

