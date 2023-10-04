import os
import numpy as np 
import torch
import cv2
from .plotting import make_matching_figure
import matplotlib.cm as cm
from tqdm import tqdm
from pathlib import Path
def show_match(img0_raw, img1_raw, mkpts0, mkpts1,mconf, save_path):
    mkpts0 = mkpts0.cpu().numpy()
    mkpts1 = mkpts1.cpu().numpy()
    mconf = mconf.cpu().numpy()
    color = cm.jet(mconf)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, path = save_path)



def main(data, matcher, save_loc_path):
    pairs = data["pairs"]
    pbar = tqdm(total=len(pairs), unit='pts')
    match_res = {}
    index_test = 0
    for imgq_pth, render_list in tqdm(pairs.items()):
        imgq_pth = str(imgq_pth)   
        max_correct = 0
        imgq_name = os.path.basename(imgq_pth)

        for render_pth in render_list[index_test*15: (index_test+1)*15]:
            imgr_pth = render_pth[0]
            exrr_pth = render_pth[1]
            imgr_name = os.path.basename(imgr_pth)
            
            imgr_pth = str(imgr_pth) 
            if imgr_pth.split('.')[-1] != 'jpg':
                imgr_pth = imgr_pth.split('.')[0] + '.jpg'
            
            matches, _, _, mconf = matcher(imgq_pth, imgr_pth)   
            F, mask = cv2.findFundamentalMat(matches[:,:2], matches[:,2:], cv2.FM_RANSAC,3, 0.99)
            index = np.where(mask == 1)
            new_matches = matches[index[0]]
            mconf = torch.tensor(mconf[index[0]]).float()
            mkpts0q = torch.tensor(new_matches[:,:2]).float()
            mkpts1r = torch.tensor(new_matches[:,2:]).float()
            correct = torch.tensor(mkpts0q.shape[0])
#============visual
            if correct > max_correct:
                max_correct = correct
                imgr_name_final = imgr_name
                imgr_pth_final = imgr_pth
                exrr_pth_final = exrr_pth
                mkpts1r_final = mkpts1r
                mkpts0q_final = mkpts0q
                mconf_final = mconf
        # if not os.path.exists(save_loc_path / 'matches'):
        #     os.makedirs(save_loc_path/ 'matches/')
        # match_vis_path = save_loc_path/ 'matches'/  (imgq_name.split('.')[0] + '.png')
        # imgq_raw = cv2.imread(str(imgq_pth), cv2.IMREAD_GRAYSCALE)
        # imgr_raw = cv2.imread(str(imgr_pth_final), cv2.IMREAD_GRAYSCALE)
        # show_match(imgq_raw,imgr_raw,mkpts0q_final,mkpts1r_final,mconf_final,match_vis_path) 
        index_test += 1 
        pbar.update(1)       
        match_res[imgq_name] = {
            "imgr_name": imgr_name_final,
            "exrr_pth": exrr_pth_final,
            "mkpts_r": mkpts1r_final,
            "mkpts_q": mkpts0q_final,
            "correct": max_correct
            }
    data["matches"] = match_res
    pbar.close() 

    return data
        
        
        