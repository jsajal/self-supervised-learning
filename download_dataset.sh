#!/bin/bash
# script for downloading the dataset

cd data
wget https://challenge.kitware.com/api/v1/item/5ac37a9d56357d4ff856e176/download
wget https://challenge.kitware.com/api/v1/item/5ac3695656357d4ff856e16a/download

unzip download
unzip download.1

rm download
rm download.1

cd ISIC2018_Task1-2_Training_Input
p=1
for i in *.jpg;
do
	mv "$i" "ISIC_Input_$p.jpg"
	((p++));
done
cd ..

cd ISIC2018_Task1_Training_GroundTruth
p=1
for i in *.png;
do
	mv "$i" "ISIC_Mask_$p.png"
	((p++));
done
cd ..

cd ..
