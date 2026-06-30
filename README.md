## EBMPose
Code for "Learning Energy-Based Models for 3D Human Pose Estimation"

This work addresses the limitation of MSE-based 3D human pose estimation methods, which assume a fixed-variance Gaussian distribution and thus have limited expressiveness. We propose a conditional energy-based model (EBM) that learns an energy function between 2D and 3D joints and refines the initial 3D pose via gradient-based optimization. 

- Paper: https://ieeexplore.ieee.org/document/10650155

### Dependencies

- Cudatoolkit: 10.2
- Python: 3.7.11
- Pytorch: 1.10.0 

Create conda environment:
```bash
conda env create -f environment.yml
```

### Dataset setup

You can obtain the Human3.6M dataset from the [Human3.6M](http://vision.imar.ro/human3.6m/) website, and then set it up using the instructions provided in [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). 

You also can access the processed data by downloading it from [here](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC?usp=sharing).

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

### Evaluation and Training

To reproduce the results of the model on the Human3.6M dataset, load the energy model weights from the `checkpoint` directory and the IGANet model weights from `previous_dir`:

```bash
python main.py --reload --previous_dir './pre_trained_model' --model model_IGANet --ebm_ckpt './checkpoint/nice_ebm.pth'
```

To retrain the energy-based model, run the following command:

```bash
python gmm_train_ebm.py --gmm_stds 0.05 0.1 0.2
```
