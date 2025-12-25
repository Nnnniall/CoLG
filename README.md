# CoLG
This is the official Pytorch implementation of our paper:
> **Co-teaching with Local and Global Information for Weakly Supervised Video Anomaly Detection**
> 
> Chen Xu, Chunguo Li, Hongjie Xing
- **Accepted at IJCNN 2025.**

![framework](data/framework.png)

## Setup

**We use the extracted I3D features for UCF-Crime and XD-Violence datasets from the following works:**

> [**UCF-Crime 10-crop I3D features**](https://github.com/tianyu0207/RTFM?tab=readme-ov-file)
> 
> [**XD-Violence 5-crop I3D features**](https://roc-ng.github.io/XD-Violence/)
>
> [**best performance ckpt for UCF-Crime**](models/DSS-ucf.pth)
>
> [**best performance ckpt for XD-Violence**](models/DSS-xd.pth)

The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the download datasets above in `list/xd_train.list` and `list/xd_test.list`.
- Feel free to change the hyperparameters in `option.py`.

## Training and testing

Start the visdom for visualizing the training phase:

```
python -m visdom.server
```
Training and inference:
```
python main.py
python test.py
```

## References

We referenced the repos below for the code:

* [RTFM](https://github.com/tianyu0207/RTFM)
* [XDVioDet](https://github.com/Roc-Ng/XDVioDet)

## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@InProceedings{Xu_2025_IJCNN,
    author = {Xu, Chen and Li, Chunguo and Xing, Hongjie},
    booktitle = {2025 International Joint Conference on Neural Networks (IJCNN)}, 
    title = {Co-teaching with Local and Global Information for Weakly Supervised Video Anomaly Detection},
    year = {2025},
    pages = {1-8}
}
