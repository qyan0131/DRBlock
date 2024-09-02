# DR-Block: Convolutional Dense Reparameterization for CNN Generalization Free Improvement

## Usage

This code contains the algorithm implementation and training code of the above paper.

- Environments
  - torch
  - torchvision
  - apex (if using distributed traing)

- For fast reproduction, train a ResNet-18 architecture on CIFAR-100 dataset by sipmly typing:

    ```shell
    python train.py
    ```

- To train an ImageNet model, following:
    ```
    python train.pay --dataset imagenet --data /path/imagenet/images --batch_size 256
    ```

- To use multi-gpu training (apex required):
    ```
    python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=23333  train.py --dataset imagenet --data /path/imagenet/images --batch_size 256 --dist
    ```

- To convert a training-time DR-Block into an inference-time model (if you don't have a trained model, randomly initialized weights will be applied):
    ```
    python  inference_conversion.py
    ```

- To try other models rather than ResNet-18, please refer to `L223` of train.py and modify the model settings.

## Citation
If you fink this repo or the referenced paper useful, please consider citing our paper.

```
@ARTICLE{drblock2024QYan,
  author={Yan, Qingqing and Li, Shu and He, Zongtao and Hu, Mengxian and Liu, Chengju and Chen, Qijun},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={DR-Block: Convolutional Dense Reparameterization for CNN Generalization Free Improvement}, 
  year={2024},
  doi={10.1109/TCSVT.2024.3411804}
}
```
