# EDM-CR: Fusing Sentinel-1 and Sentinel-2 Data with Diffusion Models for Cloud Removal

## 📄 Paper

You can find the paper [here](https://www.sciencedirect.com/science/article/abs/pii/S0034425725004535).

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

Download the original SEN12MS-CR dataset from [here](https://patricktum.github.io/cloud_removal/sen12mscr/).

## 🚀 Training

**Note**: Replace the dataset path in the YAML configuration file before training.

### Start training from scratch:
```bash
python main.py --cfg_path configs/cloudremoval_256_s1s2_cond_shift_1_10.yaml --save_dir logs/
```

### Resume training from checkpoint:
```bash
python main.py --cfg_path configs/cloudremoval_256_s1s2_cond_shift_1_10.yaml --save_dir logs/ --resume logs/XXXX-XX-XX-XX-XX/ckpts/model_XXXX.pth
```

## 💡 Inference

```bash
python inference.py \
  -i /path/to/dataset \
  -o /results \
  --task cloud_removal \
  --bs 1 \
  --ckpt_path logs/XXXX-XX-XX-XX-XX/ckpts/model_XXXX.pth \
  --cfg_path configs/cloudremoval_256_s1s2_cond_shift_1_10.yaml
```

## 📝 Citation

If you find our method helpful in your research, please cite:

```bibtex
@article{cai2025fusing,
  title={Fusing Sentinel-1 and Sentinel-2 data with diffusion models for cloud removal},
  author={Cai, Jiajun and Huang, Bo and Liu, Hao},
  journal={Remote Sensing of Environment},
  volume={331},
  pages={115049},
  year={2025},
  publisher={Elsevier}
}
```

## ⚖️ License

This project is licensed under [NTU S-Lab License 1.0](https://github.com/sczhou/CodeFormer/blob/master/LICENSE). Redistribution and use should follow this license.

## 🙏 Acknowledgements

We thank the authors of the following excellent works that inspired our research:

- [SEN12MS-CR-TS](https://github.com/PatrickTUM/SEN12MS-CR-TS)
- [ResShift](https://github.com/zsyOAOA/ResShift) 
- [UnCRtainTS](https://github.com/PatrickTUM/UnCRtainTS) 
- [GLF-CR](https://github.com/xufangchn/GLF-CR)
- [DSen2-CR](https://github.com/ameraner/dsen2-cr)
- [SpA-GAN](https://github.com/Penn000/SpA-GAN_for_cloud_removal) 
- [MC-GAN](https://github.com/enomotokenji/mcgan-cvprw2017-pytorch)
