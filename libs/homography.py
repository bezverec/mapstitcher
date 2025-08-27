from libs.util import *
import torch


'''def make_correspondences(device, model='outdoor', debug=False):
    from kornia.feature                    import LoFTR # type: ignore
    from torchvision.transforms.functional import rgb_to_grayscale
    loftr = LoFTR(pretrained = model).to(device)
    loftr.eval()
    def correspondences(image0, image1):
        with torch.inference_mode():
            matches = loftr({ # The images have to fit on GPU here
                'image0': rgb_to_grayscale(image0.unsqueeze(0)).to(device),
                'image1': rgb_to_grayscale(image1.unsqueeze(0)).to(device)})
            return matches['confidence'].cpu().numpy(),[
                    matches['keypoints0'].cpu().numpy(), matches['keypoints1'].cpu().numpy()]
    if debug: # Plot side to side correspondences
          from matplotlib import pyplot
          def correspondences_debug(image0, image1):
              conf, [kp0, kp1]  = correspondences(image0, image1)
              fig , [ax0, ax1]  = pyplot.subplots(1, 2, tight_layout = True)
              line_between_axes = make_line_between_axes(fig, ax0, ax1)
              for p0, p1, c in zip(kp0, kp1, conf):
                  line_between_axes(p0, p1, color = [1, 0, 0, float(c)]) # Set alpha based on confidence
              ax0.imshow(to_numpy_image(image0)); ax0.set_title('Correspondences 0')
              ax1.imshow(to_numpy_image(image1)); ax1.set_title('Correspondences 1')
              pyplot.show()
              pyplot.savefig('homography2.png')
              return conf, [kp0, kp1]
          return correspondences_debug
    else: return correspondences'''

def make_correspondences(device, model='outdoor', algorithm='loftr', debug=False):
    if algorithm == 'loftr':
        from kornia.feature import LoFTR
        if ".ckpt" in model: 
            print("Loading LoFTR from checkpoint:", model)
            loftr = LoFTR(pretrained=None)
            checkpoint = torch.load(model, map_location=device, weights_only=True)
            loftr.load_state_dict(checkpoint['state_dict'])
            loftr = loftr.to(device)
        else:
            loftr = LoFTR(pretrained=model).to(device)
        loftr.eval()
    
    def correspondences_loftr(image0, image1):
        from torchvision.transforms.functional import rgb_to_grayscale
        with torch.inference_mode():
            matches = loftr({
                'image0': rgb_to_grayscale(image0.unsqueeze(0)).to(device),
                'image1': rgb_to_grayscale(image1.unsqueeze(0)).to(device)})
            return matches['confidence'].cpu().numpy(), [
                    matches['keypoints0'].cpu().numpy(), matches['keypoints1'].cpu().numpy()]

    def correspondences_sift(image0, image1):
        from cv2 import SIFT_create, cvtColor, COLOR_RGB2GRAY, BFMatcher, NORM_L2
        from numpy import array, argsort, uint8
        sift = SIFT_create()
        
        img0 = image0.permute(1, 2, 0).cpu().numpy()
        img1 = image1.permute(1, 2, 0).cpu().numpy()
        img0_gray = cvtColor(img0, COLOR_RGB2GRAY)
        img1_gray = cvtColor(img1, COLOR_RGB2GRAY)
        img0_gray = (img0_gray * 255).astype(uint8) if img0_gray.dtype != uint8 else img0_gray
        img1_gray = (img1_gray * 255).astype(uint8) if img1_gray.dtype != uint8 else img1_gray

        kp0, des0 = sift.detectAndCompute(img0_gray, None)
        kp1, des1 = sift.detectAndCompute(img1_gray, None)

        bf = BFMatcher(NORM_L2, crossCheck=True)
        matches = bf.match(des0, des1)
        matches = sorted(matches, key=lambda x: x.distance)

        kp0_pts = array([kp0[m.queryIdx].pt for m in matches])
        kp1_pts = array([kp1[m.trainIdx].pt for m in matches])
        confidences = array([1 / (1 + m.distance) for m in matches])

        return confidences, [kp0_pts, kp1_pts]

    def correspondences(image0, image1):
        if algorithm == 'loftr':
            return correspondences_loftr(image0, image1)
        elif algorithm == 'sift':
            return correspondences_sift(image0, image1)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    if debug:
        from matplotlib import pyplot

        def correspondences_debug(image0, image1):
            conf, [kp0, kp1] = correspondences(image0, image1)
            fig, [ax0, ax1] = pyplot.subplots(1, 2, tight_layout=True)
            line_between_axes = make_line_between_axes(fig, ax0, ax1)

            for p0, p1, c in zip(kp0, kp1, conf):
                line_between_axes(p0, p1, color=[1, 0, 0, float(c)])  # Set alpha based on confidence

            ax0.imshow(to_numpy_image(image0))
            ax0.set_title('Correspondences 0')
            ax1.imshow(to_numpy_image(image1))
            ax1.set_title('Correspondences 1')
            pyplot.show()

            return conf, [kp0, kp1]

        return correspondences_debug

    else:
        return correspondences


def make_homography(device, model='outdoor', algorithm='loftr', debug=False, keep_ratio=0.9):
    from cv2 import findHomography, RANSAC
    correspondences = make_correspondences(device, model, algorithm, debug)
    def homography(image0, image1):
        confidence, [kp0, kp1]  = correspondences(image0, image1)

        sorted_indices = argsort(-confidence)

        num_to_keep = (int)(len(confidence) * keep_ratio)
        top_indices = sorted_indices[:num_to_keep]

        confidence_filtered = confidence[top_indices]
        kp0_filtered = kp0[top_indices]
        kp1_filtered = kp1[top_indices]

        H, inlier_mask = findHomography(kp0_filtered, kp1_filtered, RANSAC, 4.0)
        inlier_mask    = inlier_mask.flatten()
        return H, [kp0_filtered[1 == inlier_mask], kp1_filtered[1 == inlier_mask]]
    if debug:
          from matplotlib import pyplot
          def homography_debug(image0, image1):
              H  , [kp0_filtered, kp1_filtered]      = homography(image0, image1)
              fig, [ax0, ax1, ax2] = pyplot.subplots(1, 3, tight_layout = True)
              line_between_axes    = make_line_between_axes(fig, ax0, ax1)
              for p0, p1 in zip(kp0_filtered, kp1_filtered):
                  line_between_axes(p0, p1, color = [1, 0, 0])
              ax0.imshow(to_numpy_image(image0)); ax0.set_title('Inliers 0')
              ax1.imshow(to_numpy_image(image1)); ax1.set_title('Inliers 1')
              ax2.imshow(warp_image(image0, H)) ; ax2.set_title('Warped')
              pyplot.show()
              return H, [kp0_filtered, kp1_filtered]
          return homography_debug
    else: return homography

def make_homography_with_downscaling(max_size, device, model='outdoor', algorithm='loftr', debug=False):
    """Computes homography on downscaled images to allow compute feature
    correspondences on GPU with large images. The computed homography is
    corrected to describe perspective transformation on original images.  """
    from numpy import array
    resize_image = make_resize_image(max_size)
    homography   = make_homography(device, model, algorithm, debug)
    def homography_with_downscaling(image0, image1):
        small0, scale0     = resize_image(image0)
        small1, scale1     = resize_image(image1)
        H     , [kp0, kp1] = homography(small0, small1)
        '''inv_scale0         = 1 / scale0
        return [H * array([[     1,      1, inv_scale0],
                           [     1,      1, inv_scale0],
                           [scale0, scale0,          1]]),
                [kp0 * inv_scale0, kp1 * 1 / scale1]]'''
        scale_matrix0 = array([[1 / scale0, 0,         0],
                                [0,         1 / scale0, 0],
                                [0,         0,         1]])
        scale_matrix1 = array([[scale1, 0,      0],
                                [0,      scale1, 0],
                                [0,      0,      1]])
        H_corrected = scale_matrix0 @ H @ scale_matrix1
        return [H_corrected, [kp0 * (1 / scale0), kp1 * (1 / scale1)]]
    if debug:
          from matplotlib import pyplot
          def homography_with_downscaling_debug(image0, image1):
              H  , [kp0, kp1]      = homography_with_downscaling(image0, image1)
              fig, [ax0, ax1, ax2] = pyplot.subplots(1, 3, tight_layout = True)
              line_between_axes    = make_line_between_axes(fig, ax0, ax1)
              for p0, p1 in zip(kp0, kp1):
                  line_between_axes(p0, p1, color = [1, 0, 0])
              ax0.imshow(to_numpy_image(image0)); ax0.set_title('Image 0')
              ax1.imshow(to_numpy_image(image1)); ax1.set_title('Image 1')
              ax2.imshow(warp_image(image0, H)) ; ax2.set_title('Warped')
              pyplot.show()
              return H, [kp0, kp1]
          return homography_with_downscaling_debug
    else: return homography_with_downscaling

def make_affine_transform(device, model='outdoor', algorithm='loftr', debug=False, keep_ratio=0.9):
    correspondences = make_correspondences(device, model, algorithm, debug)
    def affine_transform(image0, image1):
        from cv2 import estimateAffine2D, RANSAC
        from numpy import vstack, argsort
        confidence, [kp0, kp1] = correspondences(image0, image1)

        sorted_indices = argsort(-confidence)

        num_to_keep = (int)(len(confidence) * keep_ratio)
        top_indices = sorted_indices[:num_to_keep]

        confidence_filtered = confidence[top_indices]
        kp0_filtered = kp0[top_indices]
        kp1_filtered = kp1[top_indices]

        A, inlier_mask = estimateAffine2D(kp0_filtered, kp1_filtered, method=RANSAC, ransacReprojThreshold=4.0)
        if A is not None:
            A_3x3 = vstack([A, [0, 0, 1]])
        else:
            raise ValueError("Affine transform could not be estimated.")
        inlier_mask = inlier_mask.flatten()
        return A_3x3, [kp0_filtered[1 == inlier_mask], kp1_filtered[1 == inlier_mask]]
    if debug:
        from matplotlib import pyplot
        def affine_transform_debug(image0, image1):
            A, [kp0_filtered, kp1_filtered] = affine_transform(image0, image1)
            fig, [ax0, ax1, ax2] = pyplot.subplots(1, 3, tight_layout=True)
            line_between_axes = make_line_between_axes(fig, ax0, ax1)
            for p0, p1 in zip(kp0_filtered, kp1_filtered):
                line_between_axes(p0, p1, color=[1, 0, 0])
            ax0.imshow(to_numpy_image(image0)); ax0.set_title('Inliers 0')
            ax1.imshow(to_numpy_image(image1)); ax1.set_title('Inliers 1')
            ax2.imshow(warp_image(image0, A)) ; ax2.set_title('Warped')
            pyplot.show()
            return A, [kp0_filtered, kp1_filtered]
        return affine_transform_debug
    else:
        return affine_transform

def make_affine_transform_with_downscaling(max_size, device, model='outdoor', algorithm='loftr', debug=False):
    resize_image = make_resize_image(max_size)
    affine_transform = make_affine_transform(device, model, algorithm, debug)
    
    def affine_transform_with_downscaling(image0, image1):
        from numpy import array
        small0, scale0 = resize_image(image0)
        small1, scale1 = resize_image(image1)
        
        A, [kp0, kp1] = affine_transform(small0, small1)
        if A.shape == (2, 3):
            A_3x3 = vstack([A, [0, 0, 1]])
        else:
            A_3x3 = A
        
        scale_matrix0 = array([[1 / scale0, 0,         0],
                               [0,         1 / scale0, 0],
                               [0,         0,         1]])
        scale_matrix1 = array([[scale1, 0,      0],
                               [0,      scale1, 0],
                               [0,      0,      1]])
        
        A_corrected = scale_matrix0 @ A_3x3 @ scale_matrix1
        
        kp0_corrected = kp0 * (1 / scale0)
        kp1_corrected = kp1 * (1 / scale1)
        
        return [A_corrected, [kp0_corrected, kp1_corrected]]
    
    if debug:
        from matplotlib import pyplot
        def affine_transform_with_downscaling_debug(image0, image1):
            A, [kp0, kp1] = affine_transform_with_downscaling(image0, image1)
            fig, [ax0, ax1, ax2] = pyplot.subplots(1, 3, tight_layout=True)
            line_between_axes = make_line_between_axes(fig, ax0, ax1)
            for p0, p1 in zip(kp0, kp1):
                line_between_axes(p0, p1, color=[1, 0, 0])
            ax0.imshow(to_numpy_image(image0)); ax0.set_title('Image 0')
            ax1.imshow(to_numpy_image(image1)); ax1.set_title('Image 1')
            ax2.imshow(warp_image(image0, A)) ; ax2.set_title('Warped')
            pyplot.show()
            return A, [kp0, kp1]
        return affine_transform_with_downscaling_debug
    else:
        return affine_transform_with_downscaling

if '__main__' == __name__:
    """Run as python -m libs.homography from a command line."""
    from pprint import pprint
    homography = make_homography_with_downscaling(**{  # Expects following argparse arguments.
        'max_size': 128, # Homography is computed on images of this size
        'device'  : 'cuda',
        'debug'   : True})
    pprint(homography(
        read_image('test_data/2610465282_cut_part_1.tif.jpg'),
        read_image('test_data/2610465282_cut_part_2.tif.jpg')))
