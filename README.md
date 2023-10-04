## Render-and-Compare: Cross-view 6-DoF Localization from Noisy Prior
This repository contains the official implementation of the Render2loc localization algorithm. 

Please make sure you have access to the **AirLoc dataset** and have set it up properly before proceeding. 

<p align="center">
  <img src="assets/full_img.png" height="500">
</p>

The Render2Loc localization algorithm is officially presented in the paper accepted to ICME 2022
<br>
**Render-and-Compare: Cross-view 6-DoF Localization from Noisy Prior**
<br>
Links: [arXiv](https://arxiv.org/pdf/2302.06287.pdf) | [code repos](https://github.com/) 

##  Get started

### Installation
you can refer to [hloc](https://github.com/cvg/Hierarchical-Localization/), which provides some installation guide ,and you have to install immatch. please refer to  [immatch](https://github.com/GrumpyZhou/image-matching-toolbox/blob/main/docs/install.md). You'll also need to download [blender](https://www.blender.org/download/), a rendering tool.

```bash
python3 -m venv venvrender2loc
source venvrender2loc/bin/activate
pip3 install pip -U && pip3 install -r setup/requirements.txt
```

## How To Run?
### Data Preparation
1. We will provide our AirLoc dataset, you can download from [baidu cloud](https://pan.baidu.com/s/1iWi8iGK61J_hvOAD2ofQEw) password:egkh. And the data structure is as follows. 
```bash
-- data # (you should make new dir to put all dataset)
    -- 3D Model 
    -- Query
        -- image
            -- phone
                -- day
                -- night
            -- UAV
                -- day
                -- night
        -- queries_gt # (pose gt .txt)
        -- queries_intrinsics # (intrinsic.txt)
        -- queries_prior # (pose prior.txt)

```

### Performance Testing
```bash
python3 -m pipeline.pipeline_image
```

# Citation

If you find our code useful for your research, please cite the paper:
````bibtex
@article{yan2023render,
  title={Render-and-Compare: Cross-View 6 DoF Localization from Noisy Prior},
  author={Yan, Shen and Cheng, Xiaoya and Liu, Yuxiang and Zhu, Juelin and Wu, Rouwan and Liu, Yu and Zhang, Maojun},
  journal={arXiv preprint arXiv:2302.06287},
  year={2023}
}
````