
## 19.04.02


ylaunch --cpu=5 --memory=50000 --gpu=1 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --display_id -1  --model st_gan --netD fc --batch_size 1 --num_threads 4  --crop_size 128 --load_size 128  --name paste-l2_0.001-stn_w_1-bn_1 --l2 0.001 --stn_w 1 --print_freq 10000



ylaunch --cpu=13 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --display_id -1  --model st_gan --netD fc --batch_size 240 --num_threads 25  --crop_size 128 --load_size 128  --name paste-l2_0.001-stn_w_1-bn_30 --l2 0.001 --stn_w 1 --print_freq 10000



ylaunch --cpu=13 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --display_id -1  --model st_gan --netD fc --batch_size 240 --num_threads 25  --crop_size 128 --load_size 128  --name paste-l2_0.0001-stn_w_1-bn_30 --l2 0.0001 --stn_w 1 --print_freq 10000


 
ylaunch --cpu=13 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --display_id -1  --model st_gan --netD fc --batch_size 240 --num_threads 25  --crop_size 128 --load_size 128  --name paste-l2_0.00033-stn_w_1-bn_30 --l2 0.00033 --stn_w 1 --print_freq 10000

 



 
 
 

## befor 19.04.01
 ylaunch --cpu=12 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/glasses_wear  --model cycle_gan --display_id -1 --batch_size 24 --num_threads 25 --l2 0.1 --name l2_0.1
 
 ylaunch --cpu=12 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/glasses_wear  --model cycle_gan --display_id -1 --batch_size 24 --num_threads 25 --l2 0.001 --name l2_0.001

 ylaunch --cpu=12 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/glasses_wear  --model cycle_gan --display_id -1 --batch_size 24 --num_threads 25 --l2 0.0001 --name l2_0.0001-stn_w_10  --stn_w 10

 ylaunch --cpu=12 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/glasses_wear  --model cycle_gan --display_id -1 --batch_size 24 --num_threads 25 --l2 0.001 --name l2_0.001-stn_w_10  --stn_w 10
