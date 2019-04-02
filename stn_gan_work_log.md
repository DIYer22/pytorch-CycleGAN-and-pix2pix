
## 19.04.02

 ylaunch --cpu=12 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --model st_gan --display_id -1 --batch_size 240 --num_threads 25 --name stgan-l2_0.001-stn_w_1 --l2 0.001 --stn_w 1
 
 
 ylaunch --cpu=12 --memory=50000 --gpu=1 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --model st_gan --display_id -1 --batch_size 16 --num_threads 25 --name stgan-l2_0.001-stn_w_1 --l2 0.001 --stn_w 1
 
 

## befor 19.04.01
 ylaunch --cpu=12 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/glasses_wear  --model cycle_gan --display_id -1 --batch_size 24 --num_threads 25 --l2 0.1 --name l2_0.1
 
 ylaunch --cpu=12 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/glasses_wear  --model cycle_gan --display_id -1 --batch_size 24 --num_threads 25 --l2 0.001 --name l2_0.001

 ylaunch --cpu=12 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/glasses_wear  --model cycle_gan --display_id -1 --batch_size 24 --num_threads 25 --l2 0.0001 --name l2_0.0001-stn_w_10  --stn_w 10

 ylaunch --cpu=12 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/glasses_wear  --model cycle_gan --display_id -1 --batch_size 24 --num_threads 25 --l2 0.001 --name l2_0.001-stn_w_10  --stn_w 10
