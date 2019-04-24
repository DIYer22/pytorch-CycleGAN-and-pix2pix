## 19.04.24
1. 通过 W 复合线性变换拿到了两眼的 GT
2. avgPooling for numpy image

计划:  
 * 写评价指标
 * 添加 Tensorboard
 * 设置两眼附近 变化的距离 loss, 考虑是两个点还是三个点的距离loss
 * PG-ST-GAN: 网络结构, 损失函数, loss 权重, 训练逻辑
 
长线计划:
 * 只生成单个墨镜的图像 -> 生成所有眼睛 -> 真实 celebA 眼睛 -> 根据性别自动挑选眼睛 -> 银镜和帽子联合训练 -> 自动分割眼镜 mask 




## 19.04.03
1. new data: disturbanceImg(img,15,0,0.1)
2. add theta_w
3. change GAP to fc(32*7*7)


ylaunch --cpu=13 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --print_freq 10000 --display_id -1  --model st_gan --netD fc --batch_size 240 --num_threads 25  --crop_size 128 --load_size 128  --l2 0.0001 --stn_w 1 --name 0403-l2_0.0001


ylaunch --cpu=13 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --print_freq 10000 --display_id -1  --model st_gan --netD fc --batch_size 240 --num_threads 25  --crop_size 128 --load_size 128  --l2 0.0001 --stn_w 1 --theta_w 0.1 --name 0403-l2_0.0001-theta_w_0.1


ylaunch --cpu=13 --memory=50000 --gpu=8 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --print_freq 10000 --display_id -1  --model st_gan --netD fc --batch_size 240 --num_threads 25  --crop_size 128 --load_size 128  --l2 0.0001 --stn_w 1 --theta_w 0.01 --name 0403-l2_0.0001-theta_w_0.01



ylaunch --cpu=13 --memory=50000 --gpu=4 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --print_freq 10000 --display_id -1  --model st_gan --netD fc --batch_size 240 --num_threads 25  --crop_size 128 --load_size 128  --l2 0.00001 --stn_w 1 --theta_w 0.1 --name 0403-l2_0.00001-theta_w_0.1


---

ylaunch --cpu=13 --memory=50000 --gpu=4 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --print_freq 10000 --display_id -1  --model st_gan --batch_size 240 --num_threads 25  --crop_size 128 --load_size 128  --l2 0.0001 --stn_w 1 --theta_w 0.1 --lr 0.00001  --name 0403-l2_0.0001-theta_w_0.1-lr_0.00001-no-fc


ylaunch --cpu=13 --memory=50000 --gpu=4 -- python train.py --dataroot datasets/eyeglasses_stgan_dataset  --print_freq 10000 --display_id -1  --model st_gan --batch_size 240 --num_threads 25  --crop_size 128 --load_size 128  --l2 0.00001 --stn_w 1 --theta_w 0.1 --lr 0.00001  --name 0403-l2_0.00001-theta_w_0.1-lr_0.00001-no-fc

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
