# VNN
Docker Version Implementation of MobiSys'20 Paper: Fast and Scalable In-memory Deep Multitask Learning via Neural Weight Virtualization [Original GitHub Repo](https://github.com/learning1234embed/NeuralWeightVirtualization).

## Software Install and Setup
The following command is executed under Ubuntu 18.04LTS with an NVIDIA GPU card. 

Software requirement: Make sure to have an [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) installed.

Hardware requirement: An NVIDIA GPU card that supports NVIDIA CUDA (but no need to install CUDA).

**Step 1.** Install Docker
```sh
sudo apt install docker.io
```

**Step 2.** Install NV-Docker
```sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
```

**Step 3.** Run VNN Container
```sh
sudo docker run --gpus all --name vnn -it registry.cn-hangzhou.aliyuncs.com/tinyedge/vnn:1.0-py2
```

**Step 4.** Reproduce the Results
(After entering the container...)
```sh
cd /home/vnn
```
(1) Weight-Page Matching
```sh
python weight_virtualization.py -mode=a -network_path=mnist
python weight_virtualization.py -mode=a -network_path=gsc
python weight_virtualization.py -mode=a -network_path=gtsrb
python weight_virtualization.py -mode=a -network_path=cifar10
python weight_virtualization.py -mode=a -network_path=svhn
```
(2) Weight-Page Optimization
```sh
./joint_optimization.sh
```
(3) In-Memory Execution vs. No In-Memory Execution
```sh
python in-memory_execute.py 
python baseline_execute.py
```
