# Generalized Zero-Shot Learning for Point Cloud Segmentation with Evidence-based Dynamic Calibration
This is the official repository related to "Generalized Zero-Shot Learning for Point Cloud Segmentation with Evidence-based Dynamic Calibration" (AAAI 2025, Oral)

Paper: [Proceeding](https://ojs.aaai.org/index.php/AAAI/article/view/32446)

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

Download [torch_cluster](https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl), [torch_sparse](https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.13-cp37-cp37m-linux_x86_64.whl), [torch_scatter](https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl) `.whl` files. or visit [site](https://data.pyg.org/whl/)
  ```
  pip install torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl
  pip install torch_sparse-0.6.13-cp37-cp37m-linux_x86_64.whl
  pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl
  ```

## Installation

### Build the backbone network

  * [ConvPoint](https://github.com/aboulch/ConvPoint) for S3DIS:
   ```
    cd ./3DGZSL/gzsl3d/convpoint/convpoint/knn
    python3 setup.py install --home="."
   ```
  Move the files `nearest_neighbors.cpython-37m-x86_64-linux-gnu.so` and `nearest_neighbors.py` from `convpoint/convpoint/knn/lib/python/KNN_NanoFLANN-0.0.0-py3.7-linux-x86_64.egg/` to `convpoint/convpoint/knn/lib/python/`.
  
  * [FKAConv](https://github.com/valeoai/FKAConv) for ScanNet v2:
   ```
    cd ./3DGZSL/gzsl3d/fkaconv
    pip install -ve .
   ```
  * [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) for SemanticKITTI:
   ```
    cd ./3DGZSL/gzsl3d/kpconv/cpp_wrappers
    bash ./compile_wrappers.sh
   ```

### Download the dataset 

```
./3DGZSL/
└── data/
    ├── s3dis/
    ├── scannet/
    └── semantic_kitti/
```
  * ScanNet v2
      ```
      scannet/
      ├── README.md
      ├── scannet_train.pickle
      └── scannet_test.pickle
      ```
      * Download the processed data from [this link](https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip).
      * If the above link does not work, follow [this link](https://github.com/charlesq34/pointnet2/tree/master/scannet) to process the data manually.

  * S3DIS
      ```
      s3ids/
      ├── README.md
      ├── Stanford3dDataset_v1.2_Aligned_Version/
      └── processed_data/
          ├── Area_1/
          ├── Area_2/
          └── ...
      ```
      * Download the Stanford3dDataset_v1.2_Aligned_Version from [this link](http://buildingparser.stanford.edu/dataset.html)
      * To preprocess the data, proceed as follows:
      ```
        python ./3DGZSL/gzsl3d/convpoint/examples/s3dis/prepare_s3dis_label.py --folder './3DGZSL/data/s3dis/Stanford3dDataset_v1.2_Aligned_Version' --dest './3DGZSL/data/s3dis/processed_data'
      ```
  * SemanticKITTI
      ```
      semantic_kitti/
      ├── README.md
      ├── semantic-kitti.yaml
      ├── semantic-kitti-all.yaml
      └── sequences/
          ├── 00/
          ├── 01/
          └── ...
      ```
      * Download the files [semantic-kitti.yaml](https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml) and [semantic-kitti-all.yaml](https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti-all.yaml).
      * Follow the ``semantic_kitti/README.md`` to reformat the directory structure.

### Download the pre-trained backbone parameter

Downlaod the parameters from [this link](https://drive.google.com/drive/folders/1cvoUn9NaDp1IMJU_m_bvfU5g-9fne9ED).

  * ScanNet v2
    ```
      ./3DGZSL/gzsl3d/fkaconv/examples/scannet/FKAConv_scannet_ZSL4/
      ├── PATH
      ├── logs.txt
      ├── config.yaml
      └── checkpoint.pth
      ```
  * S3DIS
    ```
      ./3DGZSL/gzsl3d/convpoint/examples/s3dis/ConvPoint_s3dis_ZSL4/
      ├── PATH
      └── state_dict.pth
      ```
  * SemanticKITTI
    ```
      ./3DGZSL/gzsl3d/kpconv/results/
      └── Log_SemanticKITTI/
          ├── PATH
          ├── parameters.txt
          └── checkpoints
              └── chkp_0250.tar
      ```

### Installation of the 3DGZSL package
```
cd E3DPC-GZSL/
bash install_gzsl3d.sh
```

## The code will be released soon.

## Citation

    @inproceedings{Kim_Kang_Lee_2025,
        author  = "Kim, Hyeonseok and Kang, Byeongkeun and Lee, Yeejin",
        year    = 2025,
        title   = "Generalized Zero-Shot Learning for Point Cloud Segmentation with Evidence-Based Dynamic Calibration",
        booktitle="Proceedings of the AAAI Conference on Artificial Intelligence", 
        pages   = "4248-4256",
    }
  
