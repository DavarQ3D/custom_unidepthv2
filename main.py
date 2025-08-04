import warnings
warnings.filterwarnings("ignore")

from custom_utils import *
from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid
from unidepth.utils.camera import Pinhole
from enum import Enum
import os, sys

class Dataset(Enum):
    IPHONE = 1
    NYU2 = 2
    KITTI = 3

#=============================================================================================================

if __name__ == "__main__":
    
    dtSet = Dataset.IPHONE
    temp = "/Users/3dsensing/Desktop/" if sys.platform == 'darwin' else "C:/Users/PC/Desktop/Temporary/"
    prefixPath = temp + "projects/custom_depthAnythingV2" 
    outdir = "./assets/outputs"
    os.makedirs(outdir, exist_ok=True)

    #--------------------- settings
    if dtSet == Dataset.IPHONE:
        inputPath = prefixPath + "/data/iphone/"
    elif dtSet == Dataset.NYU2:
        inputPath = prefixPath + "/data/nyu2_test/"
    elif dtSet == Dataset.KITTI:
        inputPath = prefixPath + "/data/kitti_variety/"
    else:
        raise ValueError("Unsupported dataset")

    #--------------------- load models

    encoder = "vits"                           # vits, vitb, vitl
    name = f"unidepth-v2-{encoder}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    # model.resolution_level = 9               # set resolution level (only V2)
    model.interpolation_mode = "bilinear"      # set interpolation mode (only V2)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(DEVICE).eval()

    #------------------ inference loop
    #------------------------------------------------------------------
    numFiles = len(os.listdir(inputPath)) // 2
    start = 0

    for idx in range(start, numFiles):


        print('\n'"========================================")
        print(f'============= sample --> {idx} =============')
        print("========================================", '\n')

        if dtSet == Dataset.IPHONE:

            rgbFileName = f"RGB_{idx:04d}.png"
            rgbPath = inputPath + rgbFileName 
            raw_image = cv2.imread(rgbPath)
            raw_image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)

            gtPath = inputPath + f"ARKit_DepthValues_{idx:04d}.txt" 
            gt = loadMatrixFromFile(gtPath)
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

        elif dtSet == Dataset.NYU2:

            rgbFileName = f"{idx:05d}_colors.png"
            rgbPath = inputPath + rgbFileName 
            raw_image = cv2.imread(rgbPath)

            gtPath = inputPath + f"{idx:05d}_depth.png"
            gt = cv2.imread(gtPath, cv2.IMREAD_UNCHANGED)
            gt = gt.astype(np.float64) / 1000.0           # scale to meters

            margin = 8   # remove white margin
            raw_image = raw_image[margin:-margin, margin:-margin, :]
            gt = gt[margin:-margin, margin:-margin]

        elif dtSet == Dataset.KITTI:

            rgbFileName = f"{idx:05d}_colors.png"
            rgbPath = inputPath + rgbFileName 
            raw_image = cv2.imread(rgbPath)

            gtPath = inputPath + f"{idx:05d}_depth.png"
            gt = cv2.imread(gtPath, cv2.IMREAD_UNCHANGED)
            gt = gt.astype(np.float64) / 256.0           # scale to meters

        else:
            raise ValueError("Unsupported dataset")


        rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)        # convert bgr to rgb
        rgbTensor = torch.from_numpy(np.array(rgb))
        rgbTensor = rgbTensor.permute(2, 0, 1)                  # c, h, w

        # intrinsics_path = "assets/demo/intrinsics.npy"
        # # Load the intrinsics if available
        # intrinsics = torch.from_numpy(np.load(intrinsics_path)) # 3 x 3
        # # For V2, we defined camera classes. If you pass a 3x3 tensor (as above)
        # # it will convert to Pinhole, but you can pass classes from camera.py.
        # # The `Camera` class is meant as an abstract, use only child classes as e.g.:
        # from unidepth.utils.camera import Pinhole, Fisheye624
        # camera = Pinhole(K=intrinsics) # pinhole 
        # # fill in fisheye, params: fx,fy,cx,cy,d1,d2,d3,d4,d5,d6,t1,t2,s1,s2,s3,s4
        # camera = Fisheye624(params=torch.tensor([...]))
        # predictions = model.infer(rgb, camera)

        predictions = model.infer(rgbTensor)
        depth = predictions["depth"]                # Metric Depth Estimation
        xyz = predictions["points"]                 # Point Cloud in Camera Coordinate
        intrinsics = predictions["intrinsics"]      # Intrinsics Prediction







