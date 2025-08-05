import warnings
warnings.filterwarnings("ignore")

from custom_utils import *
from unidepth.models import UniDepthV2
import os, sys

#=============================================================================================================

if __name__ == "__main__":
    
    dtSet = Dataset.KITTI
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

    weightedLsq = True
    fitOnDepth = False
    k_hi = 2.5 if dtSet == Dataset.IPHONE else 3.0

    showVisuals = True

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
    totalError = 0.0
    meanErr = 0.0
    sampleWithLowestError = 0
    samplewithHighestError = 0
    minRMSE = float('inf')
    maxRMSE = float('-inf')
    sampleCounter = 0
    start = 0

    for idx in range(start, numFiles):


        print('\n'"========================================")
        print(f'============= sample --> {idx} =============')
        print("========================================", '\n')

        if dtSet == Dataset.IPHONE:

            rgbFileName = f"RGB_{idx:04d}.png"
            rgbPath = inputPath + rgbFileName 
            bgr = cv2.imread(rgbPath)
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

            gtPath = inputPath + f"ARKit_DepthValues_{idx:04d}.txt" 
            gt = loadMatrixFromFile(gtPath)
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

        elif dtSet == Dataset.NYU2:

            rgbFileName = f"{idx:05d}_colors.png"
            rgbPath = inputPath + rgbFileName 
            bgr = cv2.imread(rgbPath)

            gtPath = inputPath + f"{idx:05d}_depth.png"
            gt = cv2.imread(gtPath, cv2.IMREAD_UNCHANGED)
            gt = gt.astype(np.float64) / 1000.0           # scale to meters

            margin = 8   # remove white margin
            bgr = bgr[margin:-margin, margin:-margin, :]
            gt = gt[margin:-margin, margin:-margin]

        elif dtSet == Dataset.KITTI:

            rgbFileName = f"{idx:05d}_colors.png"
            rgbPath = inputPath + rgbFileName 
            bgr = cv2.imread(rgbPath)

            gtPath = inputPath + f"{idx:05d}_depth.png"
            gt = cv2.imread(gtPath, cv2.IMREAD_UNCHANGED)
            gt = gt.astype(np.float64) / 256.0                         # scale to meters

        else:
            raise ValueError("Unsupported dataset")


        #------------------------- model prediction

        camera = getIntrinsics(dtSet) if feedIntrinsics else None
        depth, cropped, xyz, intrinsics = handlePredictionSteps(bgr, gt, model, camera)

        #------------------------- alignment

        maxVal = 50.0 if dtSet == Dataset.KITTI else 15.0
        gtMask, gt = getValidMaskAndClipExtremes(gt, minVal=0.01, maxVal=maxVal)  

        pred = depth
        predMask, pred = getValidMaskAndClipExtremes(pred, minVal=0.01, maxVal=maxVal)
        mask = gtMask & predMask
        
        if fitOnDepth:
            scale, shift, mask = weightedLeastSquared(pred, gt, guessInitPrms=True, k_lo=0.2, k_hi=k_hi, num_iters=10, fit_shift=False, verbose=False, mask=mask)

        else:
            x = pred[mask].ravel()  
            y = gt[mask].ravel()   
            scale, shift = estimateInitialParams(x, y, fitShift=False)     

        pred = scale * pred + shift      
        
        #========================================================================
        #========================================================================

        print("Scale:", fp(scale), ", Shift:", fp(shift), '\n')

        vertConcat = True if dtSet == Dataset.KITTI else False
        visualRes, rmse = analyzeAndPrepVis(cropped, mask, gt, pred, vertConcat=vertConcat)

        if rmse < minRMSE:
            minRMSE = rmse
            sampleWithLowestError = idx
        if rmse > maxRMSE:
            maxRMSE = rmse
            samplewithHighestError = idx

        totalError += rmse
        meanErr = totalError / (sampleCounter + 1)
        sampleCounter += 1
        print("\nmean across all images so far --> RMSE =", fp(meanErr, 6))
        print("\nimage with lowest error:", sampleWithLowestError, "--> RMSE =", fp(minRMSE, 6))
        print("image with highest error:", samplewithHighestError, "--> RMSE =", fp(maxRMSE, 6))

        if showVisuals:
            if dtSet == Dataset.IPHONE:
                ssc = 2.5 if sys.platform == 'darwin' else 2
            elif dtSet == Dataset.NYU2:
                ssc = 0.6
            elif dtSet == Dataset.KITTI:
                ssc = 0.6
            else:
                raise ValueError("Unsupported dataset")
            visualRes = cv2.resize(visualRes, None, fx=ssc, fy=ssc, interpolation=cv2.INTER_CUBIC)
            displayImage("visualRes", visualRes)




