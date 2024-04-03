REM usage:
REM python train.py --data ./data/waters.yaml --out_dir './output' --arct unet --encoder resnet34 --img_sz 512 --epochs 2 --batch_size 4 --lr 0.001 --momentum 0.9 --checkpoint True  --sub_size 1.0

python train.py --data ./data/buildings.yaml --arct unet --encoder resnet34 --img_sz 256 --epochs 2 --batch_size 8 --lr 0.001 --momentum 0.9 --checkpoint True --sub_size 1.0