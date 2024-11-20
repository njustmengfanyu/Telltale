# Telltale
### This is the implementation of the NDSS 2025 paper "Try to Poison My Deep Learning Data? Nowhere to Hide Your Trajectory Spectrum!"(Telltale). All steps run on the CPU, GPU devices are not required. [Instructions](https://drive.google.com/file/d/1xbaiwDy6itqLJQSEfa49DrohtmH9FJRu/view)
Step 1: Installing the dependencies needed to run this repository.
```
pip install -r requirements.txt
```

Step 2: Runing the defense file. The example shows a complete implementation using partial backdoor under the combination of CIFAR10+ResNet18 with the trigger being BadNet (white square of size 8 x 8 in the bottom right corner of the image). The time to train the model from scratch and collect truncation losses is often of unacceptable length, so we provide truncation loss traces that can be loaded directly. In this example, we collect loss trajectories from after the 86th epoch.
```
python telltale_defense.py
```
Additional Notes: 

(1) The 'LSTM classifiers' folder stores the LSTM based autoencoder. 

(2) The 'classifier_models' folder is the standard model construction code. 

(3) The 'data_loss_trace' folder stores the truncated loss trace data. 

(4) 'autoencoder.py' is the code for training the LSTM autoencoder. 

(5) 'telltale.py' is the complete flow of our method.

### If you find our code useful, please kindly cite our work.
```
@inproceedings{gao2025try,
  title={Try to Poison My Deep Learning Data? Nowhere to Hide Your Trajectory Spectrum!},
  author={Gao, Yansong and Peng, Huaibing and Ma, Hua and Zhang, Zhi and Wang, Shuo and Holland, Rayne and Fu, Anmin and Xue, Minhui and Abbott, Derek},
  booktitle={2025 Network and Distributed System Security (NDSS) Symposium},
  year={2025},
  organization={ISOC}
}
```
