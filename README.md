# EDM-CR: Fusing Sentinel-1 and Sentinel-2 data with diffusion models for cloud removal

## :mag_right: Highlights

[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0034425725004535)

• Enhanced diffusion model fuses SAR and optical for cloud removal.

• Novel forward diffusion simulates cloud addition to guide restoration.

• Two-branch backward diffusion improves cloud removal efficiency.

• Modified LPIPS loss preserves spatial details across multiple Sentinel-2 bands.

• Cloud-removed images boost agricultural parcel segmentation using temporal context.

## :round_pushpin: Requirments
```bash
pip install -r requirements.txt
```

## :electric_plug: Data preparation

Download the original SEN12MS-CR [here](https://zenodo.org/records/5735646).

## :flashlight: Training

    python main.py --cfg_path /configs/cloudremoval_256_s1s2_cond_shift_1_10.yaml --save_dir logs/

    python main.py --cfg_path /configs/cloudremoval_256_s1s2_cond_shift_1_10.yaml --save_dir logs/ --resume logs/XXXX-XX-XX-XX-XX/ckpts/model_XXXX.pth

## :bulb: Inference

    python inference.py -i /path/to/dataset -o /results --task cloud_removal --bs 1 --ckpt_path .logs/XXXX-XX-XX-XX-XX/ckpts/model_XXXX.pth --cfg_path /configs/cloudremoval_256_s1s2_cond_shift_1_10.yaml

## :wrench: Citation 

If you find our method helpful in your research, please cite with:

```
@article{
title = {Fusing Sentinel-1 and Sentinel-2 data with diffusion models for cloud removal},
journal = {Remote Sensing of Environment},
volume = {331},
pages = {115049},
year = {2025},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2025.115049},
url = {https://www.sciencedirect.com/science/article/pii/S0034425725004535},
author = {Jiajun Cai and Bo Huang and Hao Liu},
}
```

## License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/sczhou/CodeFormer/blob/master/LICENSE). Redistribution and use should follow this license.

## Acknowledgements

Thanks for these awesome works: [SEN12MS-CR-TS](https://github.com/PatrickTUM/SEN12MS-CR-TS), [ResShift](https://github.com/zsyOAOA/ResShift), [UnCRtainTS](https://github.com/PatrickTUM/UnCRtainTS), [GLF-CR](https://github.com/xufangchn/GLF-CR), [dsen2-cr](https://github.com/ameraner/dsen2-cr), [SpA-GAN](https://github.com/Penn000/SpA-GAN_for_cloud_removal), [mcgan](https://github.com/enomotokenji/mcgan-cvprw2017-pytorch).
