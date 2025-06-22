from torch.utils.data import Dataset
import os
import numpy as np
from monai.data import Dataset


def get_brain_dataet(args, transform):
    data = []

    mica_path = "OASIS3/"
    for subject in os.listdir(mica_path):
        mica_sub_path = os.path.join(mica_path, subject)
        if os.path.exists(mica_sub_path) == False: continue
        image = os.path.join(mica_sub_path, "mri/brainmask.nii.gz")
        atlas = os.path.join(mica_sub_path, "mri/aparc+aseg.nii.gz")
        if os.path.isfile(image) == False: continue
        data.append({"image":image, "label": atlas, "features":np.zeros((1,138)), "radiomics": np.zeros((1,72))})

    print("num of subject:", len(data))
    
    return Dataset(data=data, transform=transform)



if __name__ == "__main__":
    get_brain_dataet()