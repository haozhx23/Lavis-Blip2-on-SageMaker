
import os,json,socket,time

import torch
from PIL import Image
import requests

import pandas as pd


def get_local_files_list(file_path):
    import pathlib
    localnvme = pathlib.Path(file_path)
    fileslist = list(set([str(i) if not i.is_dir() else '#' for i in localnvme.rglob("*")]))
    if '#' in fileslist:
        fileslist.remove('#')
    return fileslist



if __name__ == "__main__":
    ITERS = int(os.environ['ITERS'])
    BATCH_SIZE = int(os.environ['BATCH_SIZE'])
    ffilelist = get_local_files_list('/tmp/test-imgs/')
    
    os.system('mkdir -p /tmp/vecoutput/')
    
    
    from lavis.models import load_model_and_preprocess
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device)

    for itr in range(ITERS):
        imglist = []
        caplist = []
        filelist = ffilelist[0+itr*BATCH_SIZE:(itr+1)*BATCH_SIZE]
        for imgf in filelist:
            raw_image = Image.open(imgf).convert("RGB")
            caption = "an image of something"
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            text_input = txt_processors["eval"](caption)

            imglist.append(image)
            caplist.append(text_input)

        image_batch = torch.cat(imglist,0)
        sample = {"image": image_batch, "text_input": caplist}
        features_multimodal = model.extract_features(sample)
        xx = torch.mean(features_multimodal.multimodal_embeds, dim=1)
        del imglist
        del caplist
        npp = xx.cpu().numpy()
        df = pd.DataFrame(npp) #convert to a dataframe
        df.to_csv(f"/tmp/vecoutput/vec_output_{itr}.csv",index=False) #save to file        

    os.system('aws s3 cp /tmp/vecoutput/ s3://<YOUR-BUCKET-NAME>/datasets/batch-emb-output/ --recursive')
