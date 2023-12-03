#!/bin/bash

chmod +x ./s5cmd

pip install -e LAVIS/
pip install transformers==4.34.1

./s5cmd sync 's3://<YOUR-BUCKET-NAME>/output-models/blip-caption/2023-10-19-07-30-03/20231019073/checkpoint_100.pth' '/tmp/tuned_ckpt/'

cp -f blip2_coco.yaml LAVIS/lavis/configs/models/blip2/blip2_coco.yaml

./s5cmd 's3://<YOUR-BUCKET-NAME>/datasets/coco2014/*' '/tmp/test-imgs/'

python batchinf.py