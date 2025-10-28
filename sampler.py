import os, sys, math, random

import numpy as np
from pathlib import Path
from contextlib import nullcontext

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist

from datapipe.datasets import create_dataset

class BaseSampler:
    def __init__(
            self,
            configs,
            use_amp=True,
            padding_offset=16,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.seed = seed
        self.use_amp = use_amp
        self.padding_offset = padding_offset

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            rank = 0
            torch.cuda.set_device(rank)

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str, flush=True)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        self.freeze_model(model)
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class EDMCRSampler(BaseSampler):
    def sample_func(self, y0, aux=None, noise_repeat=False, noise=None):
        '''
        Input:
            y0: n x c x h x w torch tensor, cloudy image, [-1, 1]
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1]
        '''
        if noise_repeat:
            self.setup_seed()

        offset = self.padding_offset
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % offset == 0 and ori_w % offset == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / offset)) * offset - ori_h
            pad_w = (math.ceil(ori_w / offset)) * offset - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        if self.configs.model.params.cond_cloudy:
            model_kwargs={
                    'cloudy':y0,
                    }
            if aux is not None:
                model_kwargs['s1'] = aux
            
        results = self.base_diffusion.p_sample_loop(
                y=y0,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=noise,
                noise_repeat=noise_repeat,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                )  

        if flag_pad:
            results = results[:, :, :ori_h, :ori_w]

        return results.clamp_(-1.0, 1.0)

    def inference(self, in_path, out_path, bs=1, noise_repeat=False):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for cloudy image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''
        def _process_per_image(im_cloudy_tensor, im_sar_tensor):
            '''
            Input:
                im_cloudy_tensor: b x c x h x w, torch tensor, [-1, 1]
                im_sar_tensor: b x 2 x h x w, torch tensor, [-1, 1]
            Output:
                im_clean_tensor: h x w x c, torch tensor, [0,1]
            '''
            context = torch.cuda.amp.autocast if self.use_amp else nullcontext
  
            with context():
                im_clean_tensor = self.sample_func(
                        im_cloudy_tensor,
                        aux=im_sar_tensor,
                        noise_repeat=noise_repeat,
                        )     # 1 x c x h x w, [-1, 1]

            im_clean_tensor = im_clean_tensor * 0.5 + 0.5

            return im_clean_tensor

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if self.rank == 0:
            assert in_path.exists()
            if not out_path.exists():
                out_path.mkdir(parents=True)

        if self.num_gpus > 1:
            dist.barrier()

        data_config = self.configs.data.test

        dataset = create_dataset(data_config)
        self.write_log(f'Find {len(dataset)} images in {in_path}')
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                shuffle=False,
                drop_last=False,
                )
        for data in dataloader:
            micro_batchsize = math.ceil(bs / self.num_gpus)
            ind_start = self.rank * micro_batchsize
            ind_end = ind_start + micro_batchsize
            micro_data = {key:value[ind_start:ind_end] for key,value in data.items()}
            if micro_data['cloudy'].shape[0] > 0:
                results = _process_per_image(
                        micro_data['cloudy'].cuda(),
                        micro_data['s1'].cuda() if 's1' in micro_data else None,
                        )    # b x h x w x c, [0, 1]

                for jj in range(results.shape[0]):
                    im_name = data['path'][1][jj].split('/')[-1].split('.')[0]
                    im_np = results[jj].squeeze(0).cpu().permute(1,2,0).clamp_(0.0, 1.0).numpy()
                    im_np = im_np*10000
                    util_image.imwrite_rs(im_np, data['path'][1][0], im_name, out_path)

        if self.num_gpus > 1:
            dist.barrier()

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

if __name__ == '__main__':
    pass

