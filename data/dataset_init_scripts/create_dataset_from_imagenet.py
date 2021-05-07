"""
python create_dataset_from_imagenet.py > create_dataset_from_imagenet.sh && sh create_dataset_from_imagenet.sh
"""

import os


IMAGE_NET_PATH = '/data/weixin/data/imagenet'
GROUP_TESTING_DATASET_PATH = '/data/weixin/data/GroupTestingDataset'


# firearm now 
firearm_list = [
    'n02749479', # assault rifle
    'n04086273', # revolver
    'n04090263', # rifle # 
] # and we will pick 50 images 

### remove classes like holster that introduce noise. 
ban_list = [
    'n03527444', # holster 
    'n02950826', # cannon
    'n02879718', # bow
    'n04552348', # warplane, military plane
    'n04347754', # submarine, pigboat, sub, U-boat
    'n02687172', # aircraft carrier, carrier, flattop, attack aircraft carrier
    'n04389033', # tank, army tank, armored combat vehicle, armoured combat vehicle
    'n03763968', # military_uniform
    'n02916936', # bulletproof_vest
    'n03929855', # pickelhaube
    'n04141327', # scabbard
    'n03954731', # plane, carpenter's plane, woodworking plane
    'n03498962', # hatchet
    'n03773504', # missile 
    'n04008634', # projectile, missile # 
    'n03000684', # chain saw, chainsaw
    'n02804610', # bassoon
    'n04336792', # stretcher
    'n03372029', # flute
    'n04485082', # tripod
    'n02704792', # amphibian, amphibious vehicle
]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

folder_list = get_immediate_subdirectories(f'{IMAGE_NET_PATH}/train') # 1k classes 

task_list = [
    firearm_list, # firearm classes
    [ f for f in folder_list if f not in ban_list and f not in firearm_list ] # 976 non-firearm classes 
]
print("echo \"number of non-firearm classes: {} \" ".format(len(task_list[1])) )


print(f"rm -rf {GROUP_TESTING_DATASET_PATH}") 
print(f"mkdir -p {GROUP_TESTING_DATASET_PATH}") 
for task_idx, task in enumerate(task_list):

    print(f"mkdir -p {GROUP_TESTING_DATASET_PATH}/{task_idx}/")
    print(f"mkdir -p {GROUP_TESTING_DATASET_PATH}/{task_idx}/train/")
    print(f"mkdir -p {GROUP_TESTING_DATASET_PATH}/{task_idx}/val/")
    for class_str in task:
        print(f"ln -s {IMAGE_NET_PATH}/train/{class_str}  {GROUP_TESTING_DATASET_PATH}/{task_idx}/train/{class_str}")
        print(f"ln -s {IMAGE_NET_PATH}/val/{class_str}  {GROUP_TESTING_DATASET_PATH}/{task_idx}/val/{class_str}")
    pass 

