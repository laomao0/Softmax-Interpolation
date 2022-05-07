# Softmax-Interpolation
My implementation of softmax interpolation ["Softmax Splatting for Video Frame Interpolation"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf).

My env: Pytorch 1.3, One GPU RTX2080Ti

Train this model SoftSplatModel_v3 on Vimeo90K 3-frames dataset.

Train logs are saved in ./logs

Model weights are saved in ./ckpt

Evaluation on Vimeo90K teset set: 35.92dB

Difference with original paper:
1. I compute the warped images using forward warp flow + backwarp image, refer to Line 158 in SoftSplatModel_v3.py
2. I use L1 loss for training


