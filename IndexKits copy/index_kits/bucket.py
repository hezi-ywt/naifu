from __future__ import annotations

import bisect
import json
import random
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union

import numpy as np
from tqdm import tqdm
from PIL import Image

from .indexer import ArrowIndexV2, IndexV2Builder


class Resolution(object):
    def __init__(self, size, *args):
        if isinstance(size, str):
            if 'x' in size:
                size = size.split('x')
                size = (int(size[0]), int(size[1]))
            else:
                size = int(size)
        if len(args) > 0:
            size = (size, args[0])
        if isinstance(size, int):
            size = (size, size)

        self.h = self.height = size[0]
        self.w = self.width = size[1]
        self.r = self.ratio = self.height / self.width

    def __getitem__(self, idx):
        if idx == 0:
            return self.h
        elif idx == 1:
            return self.w
        else:
            raise IndexError(f'Index {idx} out of range')

    def __str__(self):
        return f'{self.h}x{self.w}'


class ResolutionGroup(object):
    def __init__(self, base_size, step, align, target_ratios=None, enlarge=1, data=None):
        self.enlarge = enlarge
        self.align = align
        self.step = step
        
        if data is not None:
            self.data = data
            mid = len(self.data) // 2
            self.base_size = self.data[mid].h
            self.step = self.data[mid - 1].h - self.data[mid].h
        else:
            self.base_size = base_size
            assert base_size % align == 0, f'base_size {base_size} is not divisible by align {align}'
            if base_size is not None and not isinstance(base_size, int):
                raise ValueError(f'base_size must be None or int, but got {type(base_size)}')
            if step is None and target_ratios is None:
                raise ValueError(f'Either step or target_ratios must be provided')
            if step is not None and step > base_size // 2:
                raise ValueError(f'step must be smaller than base_size // 2, but got {step} > {base_size // 2}')

            self.data = self.calc(target_ratios)

        self.ratio = np.array([x.ratio for x in self.data])
        self.attr = ['' for _ in range(len(self.data))]
        self.prefix_space = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        prefix = self.prefix_space * ' '
        prefix_close = (self.prefix_space - 4) * ' '
        res_str = f'ResolutionGroup(base_size={self.base_size}, step={self.step}, data='
        attr_maxlen = max([len(x) for x in self.attr] + [5])
        res_str += f'\n{prefix}ID: height width   ratio {" " * max(0, attr_maxlen - 4)}count  h/16 w/16    tokens\n{prefix}'
        res_str += ('\n' + prefix).join([f'{i:2d}: ({x.h:4d}, {x.w:4d})  {self.ratio[i]:.4f}  {self.attr[i]:>{attr_maxlen}s}  '
                                         f'({x.h // 16:3d}, {x.w // 16:3d})  {x.h // 16 * x.w // 16:6d}'
                                         for i, x in enumerate(self.data)])
        res_str += f'\n{prefix_close})'
        return res_str

    @staticmethod
    def from_list_of_hxw(hxw_list):
        data = [Resolution(x) for x in hxw_list]
        data = sorted(data, key=lambda x: x.ratio)
        return ResolutionGroup(None, data=data)

    def calc(self, target_ratios=None):
        if target_ratios is None:
            return self._calc_by_step()
        else:
            return self._calc_by_ratio(target_ratios)

    def _calc_by_ratio(self, target_ratios):
        resolutions = []
        for ratio in target_ratios:
            if ratio == '1:1':
                reso = Resolution(self.base_size, self.base_size)
            else:
                hr, wr = map(int, ratio.split(':'))
                x = int((self.base_size ** 2 * self.enlarge // self.align // self.align / (hr * wr)) ** 0.5)
                height = x * hr * self.align
                width = x * wr * self.align
                reso = Resolution(height, width)
            resolutions.append(reso)

        resolutions = sorted(resolutions, key=lambda x_: x_.ratio)

        return resolutions

    def _calc_by_step(self):
        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [Resolution(self.base_size, self.base_size)]

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break

            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(Resolution(cur_height, cur_width))

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break

            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(Resolution(cur_height, cur_width))

        resolutions = sorted(resolutions, key=lambda x: x.ratio)

        return resolutions


class MultiBaseResolutionGroup(object):
    """支持多个base size的分辨率组"""
    def __init__(self, base_sizes=None, step=None, align=1, target_ratios=None, enlarge=1):
        self.enlarge = enlarge
        self.align = align
        self.step = step
        
        # 修复：确保base_sizes是列表
        if base_sizes is not None and not isinstance(base_sizes, list):
            base_sizes = [base_sizes]
            
        self.base_sizes = sorted(base_sizes) if base_sizes is not None else []
        
        # 验证所有base_size都可以被align整除
        for base_size in self.base_sizes:
            assert base_size % align == 0, f'base_size {base_size} is not divisible by align {align}'
            
        # 错误检查
        if not self.base_sizes:
            raise ValueError("base_sizes不能为空")
            
        if step is None and target_ratios is None:
            raise ValueError("Either step or target_ratios must be provided")
        
        # 为每个base_size创建分辨率组
        self.data = []
        for base_size in self.base_sizes:
            # 创建单个分辨率组并合并结果
            group = ResolutionGroup(base_size, step, align, target_ratios, enlarge)
            self.data.extend(group.data)
            
        # 按宽高比排序
        self.data = sorted(self.data, key=lambda x: x.ratio)
        
        # 现在可以安全地计算ratio
        self.ratio = np.array([x.ratio for x in self.data])
        self.attr = ['' for _ in range(len(self.data))]
        self.prefix_space = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        prefix = self.prefix_space * ' '
        prefix_close = (self.prefix_space - 4) * ' '
        res_str = f'MultiBaseResolutionGroup(base_size={self.base_sizes}, step={self.step}, data='
        attr_maxlen = max([len(x) for x in self.attr] + [5])
        res_str += f'\n{prefix}ID: height width   ratio {" " * max(0, attr_maxlen - 4)}count  h/16 w/16    tokens\n{prefix}'
        res_str += ('\n' + prefix).join([f'{i:2d}: ({x.h:4d}, {x.w:4d})  {self.ratio[i]:.4f}  {self.attr[i]:>{attr_maxlen}s}  '
                                         f'({x.h // 16:3d}, {x.w // 16:3d})  {x.h // 16 * x.w // 16:6d}'
                                         for i, x in enumerate(self.data)])
        res_str += f'\n{prefix_close})'
        return res_str

    def calc(self, target_ratios=None):
        if target_ratios is None:
            return self._calc_by_step()
        else:
            return self._calc_by_ratio(target_ratios)

    def _calc_by_ratio(self, target_ratios):
        resolutions = []
        for ratio in target_ratios:
            if ratio == '1:1':
                reso = Resolution(self.base_sizes[0], self.base_sizes[0])
            else:
                hr, wr = map(int, ratio.split(':'))
                x = int((self.base_sizes[0] ** 2 * self.enlarge // self.align // self.align / (hr * wr)) ** 0.5)
                height = x * hr * self.align
                width = x * wr * self.align
                reso = Resolution(height, width)
            resolutions.append(reso)

        resolutions = sorted(resolutions, key=lambda x_: x_.ratio)

        return resolutions

    def _calc_by_step(self):
        min_height = self.base_sizes[0] // 2
        min_width = self.base_sizes[0] // 2
        max_height = self.base_sizes[0] * 2
        max_width = self.base_sizes[0] * 2

        resolutions = [Resolution(self.base_sizes[0], self.base_sizes[0])]

        cur_height, cur_width = self.base_sizes[0], self.base_sizes[0]
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break

            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(Resolution(cur_height, cur_width))

        cur_height, cur_width = self.base_sizes[0], self.base_sizes[0]
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break

            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(Resolution(cur_height, cur_width))

        resolutions = sorted(resolutions, key=lambda x: x.ratio)

        return resolutions


class MultiIndexV2(object):
    """
    Multi-bucket index. Support multi-GPU (either single node or multi-node distributed) training.

    Parameters
    ----------
    index_files: list
        The index files.
    batch_size: int
        The batch size of each GPU. Required when using MultiResolutionBucketIndexV2 as base index class.
    world_size: int
        The number of GPUs. Required when using MultiResolutionBucketIndexV2 as base index class.
    sample_strategy: str
        The sample strategy. Can be 'uniform' or 'probability'. Default to 'uniform'.
        If set to probability, a list of probability must be provided. The length of the list must be the same
        as the number of buckets. Each probability value means the sample rate of the corresponding bucket.
    probability: list
        A list of probability. Only used when sample_strategy=='probability'.
    shadow_file_fn: callable or dict
        A callable function to map shadow file path to a new path. If None, the shadow file path will not be
        changed. If a dict is provided, the keys are the shadow names to call the function, and the values are the
        callable functions to map the shadow file path to a new path. If a callable function is provided, the key
        is 'default'.
    seed: int
        Only used when sample_strategy=='probability'. The seed to sample the indices.
    """
    buckets: List[ArrowIndexV2]

    def __init__(self,
                 index_files: List[str],
                 batch_size: Optional[int] = None,
                 world_size: Optional[int] = None,
                 sample_strategy: str = 'uniform',
                 probability: Optional[List[float]] = None,
                 shadow_file_fn: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 seed: Optional[int] = None,
                 ):
        self.buckets = self.load_buckets(index_files,
                                         batch_size=batch_size, world_size=world_size, shadow_file_fn=shadow_file_fn)

        self.sample_strategy = sample_strategy
        self.probability = probability
        self.check_sample_strategy(sample_strategy, probability)

        self.cum_length = self.calc_cum_length()

        self.sampler = np.random.RandomState(seed)
        if sample_strategy == 'uniform':
            self.total_length = sum([len(bucket) for bucket in self.buckets])
            self.ind_mapper = np.arange(self.total_length)
        elif sample_strategy == 'probability':
            self.ind_mapper = self.sample_indices_with_probability()
            self.total_length = len(self.ind_mapper)
        else:
            raise ValueError(f"Not supported sample_strategy {sample_strategy}.")

    def load_buckets(self, index_files, **kwargs):
        buckets = [ArrowIndexV2(index_file, **kwargs) for index_file in index_files]
        return buckets

    def __len__(self):
        return self.total_length

    def check_sample_strategy(self, sample_strategy, probability):
        if sample_strategy == 'uniform':
            pass
        elif sample_strategy == 'probability':
            if probability is None:
                raise ValueError(f"probability must be provided when sample_strategy is 'probability'.")
            assert isinstance(probability, (list, tuple)), \
                f"probability must be a list, but got {type(probability)}"
            assert len(self.buckets) == len(probability), \
                f"Length of index_files {len(self.buckets)} != Length of probability {len(probability)}"
        else:
            raise ValueError(f"Not supported sample_strategy {sample_strategy}.")

    def sample_indices_with_probability(self):
        ind_mapper_list = []
        accu = 0
        for bucket, p in zip(self.buckets, self.probability):
            if p == 1:
                # Just use all indices
                indices = np.arange(len(bucket)) + accu
            else:
                # Use all indices multiple times, and then sample some indices without replacement
                repeat_times = int(p)
                indices_part1 = np.arange(len(bucket)).repeat(repeat_times)
                indices_part2 = self.sampler.choice(len(bucket), int(len(bucket) * (p - repeat_times)), replace=False)
                indices = np.sort(np.concatenate([indices_part1, indices_part2])) + accu
            ind_mapper_list.append(indices)
            accu += len(bucket)
        ind_mapper = np.concatenate(ind_mapper_list)
        return ind_mapper

    def calc_cum_length(self):
        cum_length = []
        length = 0
        for bucket in self.buckets:
            length += len(bucket)
            cum_length.append(length)
        return cum_length

    def shuffle(self, seed=None, fast=False):
        if self.sample_strategy == 'probability':
            # Notice: In order to resample indices when shuffling, shuffle will not preserve the
            # initial sampled indices when loading the index.
            pass

        # Shuffle indexes
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
            random.shuffle(self.buckets)
            random.setstate(state)
        else:
            random.shuffle(self.buckets)

        self.cum_length = self.calc_cum_length()

        # Shuffle indices in each index
        for i, bucket in enumerate(self.buckets):
            bucket.shuffle(seed + i, fast=fast)

        # Shuffle ind_mapper
        if self.sample_strategy == 'uniform':
            self.ind_mapper = np.arange(self.total_length)
        elif self.sample_strategy == 'probability':
            self.ind_mapper = self.sample_indices_with_probability()
        else:
            raise ValueError(f"Not supported sample_strategy {self.sample_strategy}.")
        if seed is not None:
            sampler = np.random.RandomState(seed)
            sampler.shuffle(self.ind_mapper)
        else:
            np.random.shuffle(self.ind_mapper)

    def get_arrow_file(self, ind, **kwargs):
        """
        Get arrow file by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        arrow_file: str
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_arrow_file(ind - bias, **kwargs)

    def get_data(self, ind, columns=None, allow_missing=False, return_meta=True, **kwargs):
        """
        Get data by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        columns: str or list
            The columns to be returned. If None, return all columns.
        allow_missing: bool
            If True, omit missing columns. If False, raise an error if the column is missing.
        return_meta: bool
            If True, the resulting dict will contain some meta information:
            in-json index, in-arrow index, and arrow_name.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        data: dict
            A dict containing the data.
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_data(ind - bias, columns=columns, allow_missing=allow_missing,
                                        return_meta=return_meta, **kwargs)

    def get_attribute(self, ind, column, **kwargs):
        """
        Get single attribute by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        column: str
            The column name.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        attribute: Any
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_attribute(ind - bias, column, **kwargs)

    def get_image(self, ind, column='image', ret_type='pil', max_size=-1, **kwargs):
        """
        Get image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        column: str
            [Deprecated] The column name of the image. Default to 'image'.
        ret_type: str
            The return type. Can be 'pil' or 'numpy'. Default to 'pil'.
        max_size: int
            If not -1, resize the image to max_size. max_size is the size of long edge.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        image: PIL.Image.Image or np.ndarray
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_image(ind - bias, column, ret_type, max_size, **kwargs)

    def get_md5(self, ind, **kwargs):
        """ Get md5 by in-dataset index. """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_md5(ind - bias, **kwargs)

    def get_columns(self, ind, **kwargs):
        """ Get columns by in-dataset index. """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_columns(ind - bias, **kwargs)

    @staticmethod
    def resize_and_crop(image, target_size, resample=Image.LANCZOS, crop_type='random', is_list=False):
        """
        Resize image without changing aspect ratio, then crop the center/random part.

        Parameters
        ----------
        image: PIL.Image.Image
            The input image to be resized and cropped.
        target_size: tuple
            The target size of the image.
        resample:
            The resample method. See PIL.Image.Image.resize for details. Default to Image.LANCZOS.
        crop_type: str
            'center' or 'random'. If 'center', crop the center part of the image. If 'random',
            crop a random part of the image. Default to 'random'.

        Returns
        -------
        image: PIL.Image.Image
            The resized and cropped image.
        crop_pos: tuple
            The position of the cropped part. (crop_left, crop_top)
        """
        return ArrowIndexV2.resize_and_crop(image, target_size, resample, crop_type, is_list)


class MultiResolutionBucketIndexV2(MultiIndexV2):
    """
    Multi-resolution bucket index. Support multi-GPU (either single node or multi-node distributed) training.

    Parameters
    ----------
    index_file: str
        The index file of the bucket index.
    batch_size: int
        The batch size of each GPU.
    world_size: int
        The number of GPUs.
    shadow_file_fn: callable or dict
        A callable function to map shadow file path to a new path. If None, the shadow file path will not be
        changed. If a dict is provided, the keys are the shadow names to call the function, and the values are the
        callable functions to map the shadow file path to a new path. If a callable function is provided, the key
        is 'default'.
    """
    buckets: List['Bucket']

    def __init__(self,
                 index_file: str,
                 batch_size: int,
                 world_size: int,
                 shadow_file_fn: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 ):
        align = batch_size * world_size
        if align <= 0:
            raise ValueError(f'Align size must be positive, but got {align} = {batch_size} x {world_size}')
        
        # 检查索引文件是否存在
        if not Path(index_file).exists():
            raise FileNotFoundError(f"索引文件不存在: {index_file}")
        
        # 先尝试读取元数据，检查文件结构
        try:
            with open(index_file, 'r') as f:
                header = f.read(10000)  # 读取前10KB判断文件结构
                if '"indices_file":' in header:
                    print(f"检测到索引使用外部indices文件")
        except Exception as e:
            print(f"预检索引文件失败: {e}")
        
        # 加载桶
        self.buckets, self._resolutions = Bucket.from_multi_base_bucket_index(
            index_file,
            align=align,
            shadow_file_fn=shadow_file_fn,
        )
        
        # 检查桶列表是否为空
        if not self.buckets:
            raise ValueError(f"索引文件 {index_file} 中没有分辨率桶数据。请检查索引文件是否有效，或重新生成索引。")
        
        self.arrow_files = self.buckets[0].arrow_files
        self._base_sizes = self._resolutions.base_sizes
        self._step = self._resolutions.step
        
        self.buckets = sorted(self.buckets, key=lambda x: x.ratio)
        self.cum_length = self.calc_cum_length()
        
        self.total_length = sum([len(bucket) for bucket in self.buckets])
        assert self.total_length % align == 0, f'Total length {self.total_length} is not divisible by align size {align}'
        
        self.align_size = align
        self.batch_size = batch_size
        self.world_size = world_size
        self.ind_mapper = np.arange(self.total_length)
        
        # 检查总长度是否需要填充
        if self.total_length % align != 0:
            original_length = self.total_length
            # 计算需要填充的样本数
            padding_samples = align - (self.total_length % align)
            # 添加警告
            print(f"警告: 总长度 {self.total_length} 不是对齐大小 {align} 的整数倍")
            print(f"将自动填充 {padding_samples} 个重复样本以满足对齐要求")
            
            # 选择最后一个桶进行填充
            if self.buckets:
                last_bucket = self.buckets[-1]
                # 从桶中复制样本进行填充
                indices_to_add = last_bucket.indices[:padding_samples].tolist()
                # 将这些索引添加到桶中
                last_bucket.indices = np.append(last_bucket.indices, indices_to_add)
                # 更新总长度
                self.total_length += padding_samples
                print(f"总长度已从 {original_length} 填充至 {self.total_length}")
            else:
                # 如果没有桶（极少情况），放宽对齐要求
                print(f"没有可用的桶进行填充，暂时放宽对齐要求")
        
        # 使用断言验证或使用条件语句
        # assert self.total_length % align == 0, f'Total length {self.total_length} is not divisible by align size {align}'
        if self.total_length % align != 0:
            print(f"警告: 总长度 {self.total_length} 仍然不是对齐大小 {align} 的整数倍")
            print(f"这可能导致最后一个批次数据不完整")
        else:
            print(f"总长度 {self.total_length} 已正确对齐到 {align} 的整数倍")

    @property
    def step(self):
        return self._step

    @property
    def base_size(self):
        return self._base_sizes

    @property
    def resolutions(self):
        return self._resolutions

    def shuffle(self, seed=None, fast=False):
        # Shuffle indexes
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
            random.shuffle(self.buckets)
            random.setstate(state)
        else:
            random.shuffle(self.buckets)

        self.cum_length = self.calc_cum_length()

        # Shuffle indices in each index
        for i, bucket in enumerate(self.buckets):
            bucket.shuffle(seed + i, fast=fast)

        # Shuffle ind_mapper
        batch_ind_mapper = np.arange(self.total_length // self.batch_size) * self.batch_size
        if seed is not None:
            sampler = np.random.RandomState(seed)
            sampler.shuffle(batch_ind_mapper)
        else:
            np.random.shuffle(batch_ind_mapper)
        ind_mapper = np.stack([batch_ind_mapper + i for i in range(self.batch_size)], axis=1).reshape(-1)
        self.ind_mapper = ind_mapper

    def get_ratio(self, ind, **kwargs):
        """
        Get the ratio of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        width, height, ratio
        """
        ind = self.ind_mapper[ind]
        width, height = self.get_image(ind, **kwargs).size
        return width, height, height / width

    def get_target_size(self, ind):
        """
        Get the target size of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        target_width, target_height
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        return self.buckets[i].width, self.buckets[i].height

    def scale_distribution(self, save_file=None):
        if save_file is not None:
            scale_dict = np.load(save_file)
            for bucket in self.buckets:
                bucket.scale_dist = scale_dict[f'{bucket.height}x{bucket.width}']
        else:
            for bucket in tqdm(self.buckets):
                for index in tqdm(bucket.indices, leave=False):
                    scale = bucket.get_scale_by_index(index)
                    bucket.scale_dist.append(scale)
            scale_dict = {f'{bucket.height}x{bucket.width}': bucket.scale_dist for bucket in self.buckets}

            if save_file is not None:
                save_file = Path(save_file)
                save_file.parent.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(save_file, **scale_dict)

        return self


class MultiMultiResolutionBucketIndexV2(MultiIndexV2):
    buckets: List[MultiResolutionBucketIndexV2]

    @property
    def step(self):
        return [b.step for b in self.buckets]

    @property
    def base_size(self):
        return [b.base_size for b in self.buckets]

    @property
    def resolutions(self):
        return [b.resolutions for b in self.buckets]

    def load_buckets(self, index_files, **kwargs):
        self.batch_size = kwargs.get('batch_size', None)
        self.world_size = kwargs.get('world_size', None)
        if self.batch_size is None or self.world_size is None:
            raise ValueError("`batch_size` and `world_size` must be provided when using "
                             "`MultiMultiResolutionBucketIndexV2`.")
        buckets = [
            MultiResolutionBucketIndexV2(index_file,
                                         self.batch_size,
                                         self.world_size,
                                         shadow_file_fn=kwargs.get('shadow_file_fn', None),
                                         )
            for index_file in index_files
        ]
        return buckets

    def sample_indices_with_probability(self, return_batch_indices=False):
        bs = self.batch_size
        ind_mapper_list = []
        accu = 0
        for bucket, p in zip(self.buckets, self.probability):
            if p == 1:
                # Just use all indices
                batch_indices = np.arange(len(bucket) // bs) * bs + accu
            else:
                # Use all indices multiple times, and then sample some indices without replacement
                repeat_times = int(p)
                indices_part1 = np.arange(len(bucket) // bs).repeat(repeat_times) * bs
                indices_part2 = self.sampler.choice(len(bucket) // bs, int(len(bucket) * (p / bs - repeat_times)),
                                                    replace=False) * bs
                batch_indices = np.sort(np.concatenate([indices_part1, indices_part2])) + accu

            if return_batch_indices:
                indices = batch_indices
            else:
                indices = np.stack([batch_indices + i for i in range(bs)], axis=1).reshape(-1)
            ind_mapper_list.append(indices)
            accu += len(bucket)
        ind_mapper = np.concatenate(ind_mapper_list)
        return ind_mapper

    def shuffle(self, seed=None, fast=False):
        if self.sample_strategy == 'probability':
            # Notice: In order to resample indices when shuffling, shuffle will not preserve the
            # initial sampled indices when loading the index.
            pass

        # Shuffle indexes
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
            random.shuffle(self.buckets)
            random.setstate(state)
        else:
            random.shuffle(self.buckets)

        self.cum_length = self.calc_cum_length()

        # Shuffle indices in each index
        for i, bucket in enumerate(self.buckets):
            bucket.shuffle(seed + i, fast=fast)

        # Shuffle ind_mapper in batch level
        if self.sample_strategy == 'uniform':
            batch_ind_mapper = np.arange(self.total_length // self.batch_size) * self.batch_size
        elif self.sample_strategy == 'probability':
            batch_ind_mapper = self.sample_indices_with_probability(return_batch_indices=True)
        else:
            raise ValueError(f"Not supported sample_strategy {self.sample_strategy}.")
        if seed is not None:
            sampler = np.random.RandomState(seed)
            sampler.shuffle(batch_ind_mapper)
        else:
            np.random.shuffle(batch_ind_mapper)
        self.ind_mapper = np.stack([batch_ind_mapper + i for i in range(self.batch_size)], axis=1).reshape(-1)

    def get_ratio(self, ind, **kwargs):
        """
        Get the ratio of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        width, height, ratio
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_ratio(ind - bias, **kwargs)

    def get_target_size(self, ind):
        """
        Get the target size of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        target_width, target_height
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_target_size(ind - bias)


class MultiBaseResolutionBucketIndexV2(MultiResolutionBucketIndexV2):
    """支持多base size的多分辨率桶索引类
    
    Parameters
    ----------
    index_file: str
        索引文件路径
    batch_size: int
        每个GPU的批量大小
    world_size: int
        GPU数量
    shadow_file_fn: callable or dict
        影子文件路径映射函数
    """
    buckets: List[Bucket]
    
    def __init__(self,
                 index_file: str,
                 batch_size: int,
                 world_size: int,
                 shadow_file_fn: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 ):
        align = batch_size * world_size
        if align <= 0:
            raise ValueError(f'Align size must be positive, but got {align} = {batch_size} x {world_size}')
            
        # 检查索引文件是否存在
        if not Path(index_file).exists():
            raise FileNotFoundError(f"索引文件不存在: {index_file}")
        
        # 先尝试读取元数据，检查文件结构
        try:
            with open(index_file, 'r') as f:
                header = f.read(10000)  # 读取前10KB判断文件结构
                if '"indices_file":' in header:
                    print(f"检测到索引使用外部indices文件")
        except Exception as e:
            print(f"预检索引文件失败: {e}")
        
        # 加载桶
        self.buckets, self._resolutions = Bucket.from_multi_base_bucket_index(
            index_file,
            align=align,
            shadow_file_fn=shadow_file_fn,
        )
        
        # 检查桶列表是否为空
        if not self.buckets:
            raise ValueError(f"索引文件 {index_file} 中没有分辨率桶数据。请检查索引文件是否有效，或重新生成索引。")
        
        self.arrow_files = self.buckets[0].arrow_files
        self._base_sizes = self._resolutions.base_sizes
        self._step = self._resolutions.step
        
        self.buckets = sorted(self.buckets, key=lambda x: x.ratio)
        self.cum_length = self.calc_cum_length()
        
        self.total_length = sum([len(bucket) for bucket in self.buckets])
        assert self.total_length % align == 0, f'Total length {self.total_length} is not divisible by align size {align}'
        
        self.align_size = align
        self.batch_size = batch_size
        self.world_size = world_size
        self.ind_mapper = np.arange(self.total_length)
        
        # 检查总长度是否需要填充
        if self.total_length % align != 0:
            original_length = self.total_length
            # 计算需要填充的样本数
            padding_samples = align - (self.total_length % align)
            # 添加警告
            print(f"警告: 总长度 {self.total_length} 不是对齐大小 {align} 的整数倍")
            print(f"将自动填充 {padding_samples} 个重复样本以满足对齐要求")
            
            # 选择最后一个桶进行填充
            if self.buckets:
                last_bucket = self.buckets[-1]
                # 从桶中复制样本进行填充
                indices_to_add = last_bucket.indices[:padding_samples].tolist()
                # 将这些索引添加到桶中
                last_bucket.indices = np.append(last_bucket.indices, indices_to_add)
                # 更新总长度
                self.total_length += padding_samples
                print(f"总长度已从 {original_length} 填充至 {self.total_length}")
            else:
                # 如果没有桶（极少情况），放宽对齐要求
                print(f"没有可用的桶进行填充，暂时放宽对齐要求")
        
        # 使用断言验证或使用条件语句
        # assert self.total_length % align == 0, f'Total length {self.total_length} is not divisible by align size {align}'
        if self.total_length % align != 0:
            print(f"警告: 总长度 {self.total_length} 仍然不是对齐大小 {align} 的整数倍")
            print(f"这可能导致最后一个批次数据不完整")
        else:
            print(f"总长度 {self.total_length} 已正确对齐到 {align} 的整数倍")

    @property
    def step(self):
        return self._step
        
    @property
    def base_sizes(self):
        return self._base_sizes
        
    @property
    def resolutions(self):
        return self._resolutions
    
    def shuffle(self, seed=None, fast=False):
        # 保持与MultiResolutionBucketIndexV2相同的逻辑
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
            random.shuffle(self.buckets)
            random.setstate(state)
        else:
            random.shuffle(self.buckets)

        self.cum_length = self.calc_cum_length()

        # 对每个桶中的索引进行洗牌
        for i, bucket in enumerate(self.buckets):
            bucket.shuffle(seed + i if seed is not None else None, fast=fast)

        # 在批次级别洗牌ind_mapper
        batch_ind_mapper = np.arange(self.total_length // self.batch_size) * self.batch_size
        if seed is not None:
            sampler = np.random.RandomState(seed)
            sampler.shuffle(batch_ind_mapper)
        else:
            np.random.shuffle(batch_ind_mapper)
        self.ind_mapper = np.stack([batch_ind_mapper + i for i in range(self.batch_size)], axis=1).reshape(-1)
    
    def get_ratio(self, ind, **kwargs):
        """获取图像的宽高比"""
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        width, height = self.buckets[i].get_image(ind - bias, **kwargs).size
        return width, height, height / width
    
    def get_target_size(self, ind):
        """获取目标尺寸"""
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].width, self.buckets[i].height


def build_multi_base_resolution_bucket(config_file,
                                      base_sizes,
                                      src_index_files,
                                      save_file,
                                      reso_step=64,
                                      target_ratios=None,
                                      align=1,
                                      min_size=0,
                                      md5_hw=None,
                                      ):
    """
    构建支持多个base size的分辨率桶索引
    
    Parameters
    ----------
    base_sizes: list
        基础尺寸列表，例如[512, 768, 1024]
    """
    # 使用多base size分辨率组
    resolutions = MultiBaseResolutionGroup(base_sizes, step=reso_step, target_ratios=target_ratios, align=align)
    print(resolutions)
    
    save_file = Path(save_file)
    save_file.parent.mkdir(exist_ok=True, parents=True)
    
    if isinstance(src_index_files, str):
        src_index_files = [src_index_files]
    src_indexes = []
    print(f'Loading indexes:')
    for src_index_file in src_index_files:
        src_indexes.append(ArrowIndexV2(src_index_file))
        print(f'    {src_index_file} | cum_length: {src_indexes[-1].cum_length[-1]} | indices: {len(src_indexes[-1])}')
    
    if md5_hw is None:
        md5_hw = {}
        
    arrow_files = src_indexes[0].arrow_files[:]
    for src_index in src_indexes[1:]:
        arrow_files.extend(src_index.arrow_files[:])
        
    cum_length = src_indexes[0].cum_length[:]
    for src_index in src_indexes[1:]:
        cum_length.extend([x + cum_length[-1] for x in src_index.cum_length])
    print(f'cum_length: {cum_length[-1]}')
    
    group_length_list = src_indexes[0].group_length[:]
    for src_index in src_indexes[1:]:
        group_length_list.extend(src_index.group_length[:])
        
    total_indices = sum([len(src_index) for src_index in src_indexes])
    total_group_length = sum(group_length_list)
    assert total_indices == total_group_length, f'Total indices {total_indices} != Total group length {total_group_length}'
    
    buckets = [[] for _ in range(len(resolutions))]
    cum_length_tmp = 0
    total_index_count = 0
    
    for src_index, src_index_file in zip(src_indexes, src_index_files):
        index_count = 0
        pbar = tqdm(src_index.indices.tolist())
        for i in pbar:
            try:
                height = int(src_index.get_attribute_by_index(i, 'height'))
                width = int(src_index.get_attribute_by_index(i, 'width'))
            except Exception as e1:
                try:
                    md5 = src_index.get_attribute_by_index(i, 'md5')
                    height, width = md5_hw[md5]
                except Exception as e2:
                    try:
                        width, height = src_index.get_image_by_index(i).size
                    except Exception as e3:
                        print(f'Error: {e1} --> {e2} --> {e3}. We will skip this image.')
                        continue
                        
            if height < min_size or width < min_size:
                continue
                
            ratio = height / width
            idx = np.argmin(np.abs(resolutions.ratio - ratio))
            buckets[idx].append(i + cum_length_tmp)
            index_count += 1
        print(f"Valid indices {index_count} in {src_index_file}.")
        cum_length_tmp += src_index.cum_length[-1]
        total_index_count += index_count
    print(f'Total indices: {total_index_count}')
    
    print(f'Making bucket index.')
    indices = {}
    for i, bucket in tqdm(enumerate(buckets)):
        if len(bucket) == 0:
            continue
        reso = f'{resolutions[i]}'
        resolutions.attr[i] = f'{len(bucket):>6d}'
        indices[reso] = bucket
        
    builder = IndexV2Builder(data_type=['multi-base-resolution-bucket-v2',
                                     f'base_sizes={base_sizes}',
                                     f'reso_step={reso_step}',
                                     f'target_ratios={target_ratios}',
                                     f'align={align}',
                                     f'min_size={min_size}',
                                     f'src_files='] +
                                    [f'{src_index_file}' for src_index_file in src_index_files],
                          arrow_files=arrow_files,
                          cum_length=cum_length,
                          indices=indices,
                          config_file=config_file,
                          )
    builder.build(save_file)
    print(resolutions)
    print(f'Build index finished!\n\n'
          f'            Save path: {Path(save_file).absolute()}\n'
          f'    Number of indices: {sum([len(v) for k, v in indices.items()])}\n'
          f'Number of arrow files: {len(arrow_files)}\n'
          )


class Bucket(ArrowIndexV2):
    """分辨率桶"""
    def __init__(self, height, width, arrow_files, cum_length, indices=None, **kwargs):
        self.height = height
        self.width = width
        self.ratio = height / width
        self.scale_dist = []
        
        # 构建一个符合ArrowIndexV2预期的res_dict
        res_dict = {
            'data_type': kwargs.get('data_type', ['multi-resolution']),
            'arrow_files': arrow_files,
            'cum_length': cum_length,
            'indices': indices or [],
            'indices_file': '',
            'group_length': kwargs.get('group_length', [0] * len(arrow_files))
        }
        
        # 使用res_dict调用父类构造函数
        super().__init__(res_dict=res_dict, **kwargs)

    @classmethod
    def from_multi_base_bucket_index(cls, index_file, align=1, shadow_file_fn=None):
        """从多base size桶索引文件加载桶列表"""
        print(f"开始加载多base size索引：{index_file}")
        
        with open(index_file, 'r') as f:
            print("正在读取索引文件...")
            res_dict = json.load(f)
        
        print("解析索引文件元数据...")
        arrow_files = res_dict.get('arrow_files', [])
        cum_length = res_dict.get('cum_length', [])
        data_type = res_dict.get('data_type', [])
        
        # 检查是否使用外部indices文件
        indices = res_dict.get('indices', {})
        indices_file = res_dict.get('indices_file', None)
        
        # 如果indices为空但indices_file存在，尝试加载外部索引文件
        if not indices and indices_file:
            print(f"主索引文件没有桶数据，尝试从外部文件加载: {indices_file}")
            try:
                indices_path = Path(indices_file)
                # 如果是相对路径，相对于主索引文件
                if not indices_path.is_absolute():
                    indices_path = Path(index_file).parent / indices_path
                
                # 检查文件扩展名来决定如何加载
                if str(indices_path).endswith('.npz'):
                    print("检测到NPZ格式索引文件，使用NumPy加载")
                    # 使用binary模式和np.load加载npz文件
                    npz_data = np.load(indices_path, allow_pickle=True)
                    
                    # 输出所有键
                    print(f"NPZ文件内容键: {list(npz_data.keys())[:5]}... (共{len(npz_data.keys())}个)")
                    
                    # 查看是否有标准格式的'indices'键
                    if 'indices' in npz_data:
                        indices = npz_data['indices'].item()  # 将NumPy对象转换回Python字典
                        print(f"从NPZ文件 {indices_path} 成功加载了 {len(indices)} 个桶")
                    else:
                        # 检查是否每个键都是分辨率格式（如"256x1024"）
                        reso_keys = [k for k in npz_data.keys() if 'x' in str(k)]
                        if reso_keys:
                            print(f"检测到分辨率格式的键，直接构建indices字典")
                            indices = {}
                            # 将每个分辨率键的数据添加到indices字典中
                            for k in reso_keys:
                                # 预处理分辨率格式为(height,width)
                                h, w = map(int, str(k).split('x'))
                                reso_key = f"({h},{w})"  # 转换为标准格式
                                indices[reso_key] = npz_data[k].tolist()  # 转换NumPy数组为Python列表
                            print(f"从分辨率键构建了 {len(indices)} 个桶")
                        else:
                            print("NPZ文件不包含有效的分辨率桶数据")
                else:
                    # 默认作为JSON处理
                    with open(indices_path, 'r') as f:
                        indices = json.load(f)
                        print(f"从JSON文件 {indices_path} 成功加载了 {len(indices)} 个桶")
            except Exception as e:
                print(f"加载外部索引文件失败: {e}")
                # 尝试作为二进制文件重新加载
                try:
                    if str(indices_path).endswith('.npz'):
                        print("第二次尝试加载NPZ文件...")
                        npz_data = np.load(indices_path, allow_pickle=True)
                        print(f"NPZ文件内容键: {list(npz_data.keys())}")
                except Exception as e2:
                    print(f"二次尝试加载也失败: {e2}")
        
        print(f"索引文件包含 {len(indices)} 个不同分辨率的桶")
        
        # 尝试从data_type中获取base_sizes
        base_sizes = None
        step = None
        for dt in data_type:
            if 'base_sizes=' in str(dt):
                try:
                    base_sizes_str = str(dt).split('base_sizes=')[1].split(',')[0]
                    if base_sizes_str.startswith('[') and ']' in base_sizes_str:
                        base_sizes = eval(base_sizes_str[:base_sizes_str.index(']')+1])
                    else:
                        base_sizes = [int(x.strip()) for x in base_sizes_str.split(',')]
                except Exception as e:
                    print(f"解析base_sizes出错: {e}")
            if 'reso_step=' in str(dt):
                try:
                    step_str = str(dt).split('reso_step=')[1].split(',')[0]
                    if step_str.startswith('['):
                        step = eval(step_str[:step_str.index(']')+1])
                    else:
                        step = int(step_str)
                except Exception as e:
                    print(f"解析step出错: {e}")
        
        if base_sizes is None:
            print("警告: 无法从data_type中获取base_sizes，使用默认值[512]")
            base_sizes = [512]
            
        if step is None:
            print("警告: 无法从data_type中获取step，使用默认值64")
            step = 64
            
        print(f"创建分辨率组，base_sizes={base_sizes}, step={step}")
        resolutions = MultiBaseResolutionGroup(base_sizes, step=step, align=align)
        
        # 创建桶列表
        buckets = []
        total_count = len(indices)
        print(f"开始创建 {total_count} 个分辨率桶...")
        
        for i, (reso, bucket_indices) in enumerate(indices.items()):
            if i % 20 == 0 or i == total_count - 1:  # 每20个桶打印一次进度，以及最后一个
                print(f"处理桶 {i+1}/{total_count}...")
                
            # 调试信息
            if len(bucket_indices) == 0:
                print(f"警告: 桶 {reso} 不包含任何索引")
                continue
                
            h, w = map(int, reso.strip('()').split(','))
            bucket = cls(height=h, width=w, arrow_files=arrow_files, 
                        cum_length=cum_length, indices=bucket_indices)
            if shadow_file_fn is not None:
                bucket.shadow_file_fn = shadow_file_fn
                
            buckets.append(bucket)
        
        print(f"成功创建 {len(buckets)} 个分辨率桶")
        if len(buckets) == 0:
            print(f"警告: 没有创建任何分辨率桶! 请检查索引文件是否有效。")
        
        # 在创建桶列表后，检查并对齐总样本数
        total_samples = sum(len(bucket.indices) for bucket in buckets)
        if align > 1 and total_samples % align != 0:
            padding_needed = align - (total_samples % align)
            print(f"总样本数 {total_samples} 不是 {align} 的整数倍，需要填充 {padding_needed} 个样本")
            
            # 选择样本最多的桶进行填充
            largest_bucket = max(buckets, key=lambda b: len(b.indices), default=None)
            if largest_bucket:
                # 从最大桶复制样本进行填充
                indices_to_add = largest_bucket.indices[:padding_needed].tolist()
                # 添加到该桶的索引中
                largest_bucket.indices = np.append(largest_bucket.indices, indices_to_add)
                print(f"已添加 {padding_needed} 个重复样本到桶 ({largest_bucket.height},{largest_bucket.width})")
        
        return buckets, resolutions
