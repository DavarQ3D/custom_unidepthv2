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

    #--------------------- data
    if dtSet == Dataset.IPHONE:
        inputPath = prefixPath + "/data/iphone/"
    elif dtSet == Dataset.NYU2:
        inputPath = prefixPath + "/data/nyu2_test/"
    elif dtSet == Dataset.KITTI:
        inputPath = prefixPath + "/data/kitti_variety/"
    else:
        raise ValueError("Unsupported dataset")
    
    #--------------------- settings

    feedIntrinsics = False          

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
            gt = gt.astype(np.float64) / 256.0                         # scale to meters

        else:
            raise ValueError("Unsupported dataset")

        rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)               # convert bgr to rgb
        rgbTensor = torch.from_numpy(np.array(rgb))
        rgbTensor = rgbTensor.permute(2, 0, 1)                         # c, h, w

        camera = None
        if feedIntrinsics:
            intrinsics_path = "assets/demo/intrinsics.npy"
            intrinsics = torch.from_numpy(np.load(intrinsics_path))    # 3 x 3
            from unidepth.utils.camera import Pinhole
            camera = Pinhole(K=intrinsics) 
            print("intrinsics shape:", intrinsics.shape)
            print("intrinsics:", intrinsics)
            exit()

        predictions = model.infer(rgbTensor, camera=camera)
        
        depth = predictions["depth"]                                 # metric depth
        depth = depth[0,0].cpu().numpy()
        
        xyz = predictions["points"]                                  # point cloud in camera coordinates
        xyz = xyz[0].cpu().permute(1, 2, 0).numpy()

        intrinsics = predictions["intrinsics"]                       # intrinsics prediction
        intrinsics = intrinsics[0].cpu().numpy()

        visualRes = np.hstack([raw_image, colorize(depth)])  
        ssc = 0.4
        visualRes = cv2.resize(visualRes, None, fx=ssc, fy=ssc, interpolation=cv2.INTER_CUBIC)
        displayImage("visual", visualRes)
        
        






