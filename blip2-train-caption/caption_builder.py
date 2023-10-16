"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)

import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
import os


@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }
    
    # override method in base_dataset_builder.py
    def build_datasets(self):

        if is_main_process():
            # self._download_data()
            
            
            
            # img_s3_path = 's3://llm-artifacts-us-east-1/datasets/coco2014/'
            img_dest_path = '/tmp/lavis/coco/images/'
            
            # os.system("./s5cmd sync {0} {1}".format(img_s3_path+'*', img_dest_path))
            
            os.system("./s5cmd --log=error sync {0} {1}".format(os.environ['DATA_S3_PATH']+'*', img_dest_path))
            print(f'------rank 0 finished data-images copy-------')
            
            # anno_s3_path = 's3://llm-artifacts-us-east-1/datasets/coco2014-anno/'
            anno_dest_path = '/tmp/sub_annos/'
            
            os.system("./s5cmd sync {0} {1}".format(os.environ['ANNO_S3_PATH']+'*', anno_dest_path))
            print(f'----DDDD-----rank 0 finished img/cap copy-------')

        if is_dist_avail_and_initialized():
            dist.barrier()

        print("----DDDD-----Building datasets in COCOCapBuilder 2...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }
