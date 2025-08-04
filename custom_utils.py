import cv2
import numpy as np
from pathlib import Path

#=============================================================================================================

def fp(value, precision=4):
    return f"{value:.{precision}f}" 

def normalize(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def denormalize(image):
    return (image * 255).astype(np.uint8)

def analyzeAndPrepVis(rgb, mask, ref, pred, vertConcat=False):
    
    err = np.abs(ref - pred)
    valid = err[mask]
    rmse = np.sqrt((valid**2).mean())
    print("valid pixels RMSE =", fp(rmse, 6))

    ref = normalize(ref)
    pred = normalize(pred)

    err = np.clip(err, 0, valid.max())
    err = normalize(err) 

    ref = denormalize(ref)
    pred = denormalize(pred)    
    err = denormalize(err)

    mask = mask.astype(np.uint8) * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    err = cv2.cvtColor(err, cv2.COLOR_GRAY2BGR)
    err = cv2.applyColorMap(err, cv2.COLORMAP_JET)
    err[mask == 0] = 0                 

    visualRes = cv2.vconcat([rgb, mask, ref, pred, err]) if vertConcat else cv2.hconcat([rgb, mask, ref, pred, err])

    return visualRes, rmse

#=============================================================================================================

def displayImage(title, image):
    cv2.imshow(title, image)
    key = cv2.waitKey(0)
    if key == 27:  
        cv2.destroyAllWindows()
        exit()

#=============================================================================================================

def loadMatrixFromFile(path):
    path = Path(path)
    matrix = np.loadtxt(path, delimiter=',', dtype=np.float64)
    return matrix

#=============================================================================================================

def robustNormalize(x):
    t = np.median(x)
    s = np.mean(np.abs(x - t))
    if s == 0:
        raise ValueError("Input array has zero variance; cannot normalize.")
    normalized = (x - t) / s
    return normalized

#=============================================================================================================

def estimateInitialParams(x, y, fitShift):

    t_pred = np.median(x)
    s_pred = np.mean(np.abs(x - t_pred)) if len(x) > 0 else 1.0
    
    t_gt = np.median(y)     
    s_gt = np.mean(np.abs(y - t_gt)) if len(y) > 0 else 1.0  

    scale = s_gt / s_pred 
    shift = t_gt - scale * t_pred if fitShift else 0.0
    
    return scale, shift

#=============================================================================================================

def weightedLeastSquared(pred, gt, guessInitPrms, k_lo=0.5, k_hi=3, num_iters=5, fit_shift=True, verbose=True, mask=None):

    if mask is None:
        mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)

    if not mask.any():
        raise ValueError("No valid pixels")

    x = pred[mask].ravel()  
    y = gt[mask].ravel()    

    scale, shift = estimateInitialParams(x, y, fit_shift) if guessInitPrms else (1.0, 0.0)
    outlier_cap = None

    for iter in range(num_iters):

        fit = scale * x + shift
        residuals = y - fit
        abs_res = np.abs(residuals)

        sigma = 1.4826 * np.median(abs_res) + 1e-12
        outlier_cap = k_hi * sigma
        inlier_bottom = k_lo * sigma

        if verbose:
            print("inlier_bottom =", fp(inlier_bottom, 2), ", outlier_cap =", fp(outlier_cap, 2))
            print("iter =", iter, "-> scale =", fp(scale), ", shift =", fp(shift), ", all pixels err -> max:", fp(abs_res.max()), ", mean:", fp(abs_res.mean()), '\n')

        valid = abs_res <= outlier_cap    # hard skip outliers mask
        if not valid.any():
            break  

        abs_res = abs_res[valid]
        x_valid = x[valid]
        residuals = residuals[valid]

        weights = np.ones_like(abs_res)
        moderate = (abs_res > inlier_bottom) & (abs_res <= outlier_cap)        # moderate residuals mask in which weights aren't 1
        weights[moderate] = 1 - (abs_res[moderate] - inlier_bottom) / (outlier_cap - inlier_bottom)

        wx = weights * x_valid
        wr = weights * residuals

        if fit_shift:
            jtj = np.zeros((2, 2))
            jty = np.zeros(2)
            jtj[0, 0] = np.dot(wx, x_valid)
            jtj[0, 1] = jtj[1, 0] = np.sum(wx)
            jtj[1, 1] = np.sum(weights)
            jty[0] = np.dot(wr, x_valid)
            jty[1] = np.sum(wr)
            update = np.linalg.solve(jtj, jty)
            scale += update[0]
            shift += update[1]
        else:
            jtj_scalar = np.dot(wx, x_valid)
            jty_scalar = np.dot(wr, x_valid)
            update = 0.0 if jtj_scalar == 0 else jty_scalar / jtj_scalar
            scale += update

    final_fit = scale * pred + shift
    final_res = np.abs(gt - final_fit)
    inlier_mask = (final_res <= outlier_cap) & mask

    if verbose:
        print()

    return scale, shift, inlier_mask

#=============================================================================================================

def getValidMaskAndClipExtremes(image, minVal, maxVal):
    mask = np.isfinite(image) & (image > minVal) & (image < maxVal) 
    image = np.clip(image, minVal, maxVal)  
    return mask, image
