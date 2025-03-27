# Generalized Zero-Shot Learning for Point Cloud Segmentation with Evidence-based Dynamic Calibration
This is the official repository related to "Generalized Zero-Shot Learning for Point Cloud Segmentation with Evidence-based Dynamic Calibration" (AAAI 2025, Oral)

Meterials: [Poster](Material/E3DPC-GZSL_Poster.pdf) & [Presentation](Material/E3DPC-GZSL_presentation.pdf)

<img src="https://github.com/user-attachments/assets/c1ecabbe-065c-4fba-8844-a22db85ddcd1" alt="teaser" style="width:50%;">

## Abstract
Generalized zero-shot semantic segmentation of 3D point clouds aims to classify each point into both seen and unseen classes. A significant challenge with these models is their tendency to make biased predictions, often favoring the classes encountered during training. This problem is more pronounced in 3D applications, where the scale of the training data is typically smaller than in image-based tasks. To address this problem, we propose a novel method called E3DPC-GZSL, which reduces overconfident predictions towards seen classes without relying on separate classifiers for seen and unseen data. E3DPC-GZSL tackles the overconfidence problem by integrating an evidence-based uncertainty estimator into a classifier. This estimator is then used to adjust prediction probabilities using a dynamic calibrated stacking factor that accounts for pointwise prediction uncertainty. In addition, E3DPC-GZSL introduces a novel training strategy that improves uncertainty estimation by refining the semantic space. This is achieved by merging learnable parameters with text-derived features, thereby improving model optimization for unseen data. Extensive experiments demonstrate that the proposed approach achieves state-of-the-art performance on generalized zero-shot semantic segmentation datasets, including ScanNet v2 and S3DIS.

## Environments
* Python: 3.7
* Pytorch: 1.10.1
* CUDA: 11.3
```
conda create -n E3DPC-GZSL python=3.7 -y
conda activate E3DPC-GZSL
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

For other libraries, please refer to `requirements.txt`.
  ```
  pip install -r requirements.txt
  ```

Download [torch_cluster](https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl), [torch_sparse](https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.13-cp37-cp37m-linux_x86_64.whl), [torch_scatter](https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl) `.whl` files.
  ```
  pip install torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl
  pip install torch_sparse-0.6.13-cp37-cp37m-linux_x86_64.whl
  pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
  ```

## The code will be released soon.
