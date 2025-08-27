import numpy as np

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt
import torch.nn.functional as Fnn

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (8 - (self.ht % 8)) % 8  # Calculate the padding needed to make ht divisible by 8
        pad_wd = (8 - (self.wd % 8)) % 8  # Calculate the padding needed to make wd divisible by 8

        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        """ Apply padding to the inputs """
        return [Fnn.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        """ Remove padding from the input """
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class OpticalFlow_RAFT:
    def __init__(self, debug=False, silent=False, weights_path=None):
        """
        Initializes the OpticalFlow_RAFT class.
        """

        self.silent = silent
        self.debug = debug
        if weights_path is not None:
            print("Loading RAFT weights from", weights_path)
            self.weights = Raft_Large_Weights.DEFAULT
            self.weights._url = None
            self.weights._meta = None
            self.weights._weights_path = weights_path
            if not self.silent:
                print(f"Using RAFT weights from {weights_path}")
        else:
            self.weights = Raft_Large_Weights.DEFAULT

        self.model = raft_large(weights=self.weights)
        self.transforms = self.weights.transforms()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        # load the RAFT model

    def compute_optical_flow_tiled(self, img1, img2, orientation='horizontal', vram=8.0):
        # 1150 * 1150 per 8GB of VRAM (maybe more, but i have no means of testing)
        maxpix = (int)(1050 * 1050 * (1.0 + (vram / 8.0 - 1.0) * 0.5)) # very rough estimate
        if orientation == 'vertical':
            patch_y = maxpix // img1.shape[1]
            patch_x = img1.shape[1]
            overlap = (int)(patch_y * 0.1)

            result = np.zeros((img1.shape[0], img1.shape[1], 2), dtype=np.float32)
            weights = np.zeros((img1.shape[0], img1.shape[1], 1), dtype=np.float32)
            with torch.no_grad():
                if not self.silent:
                    print(f"Processing {img1.shape[0] // (patch_y - overlap)} flow patches {patch_y} x {patch_x}")
                for y in range(0, img1.shape[0], patch_y - overlap):
                    #print(f"Processing path {y // (patch_y - overlap)} of {img1.shape[0] // (patch_y - overlap)}")

                    # extract patches, respect image boundaries
                    img1_patch = img1[y:y+patch_y]
                    img2_patch = img2[y:y+patch_y]
                    # if patch is too small, move it up
                    if img1_patch.shape[0] < patch_y:
                        y = max(y - (patch_y - img1_patch.shape[0]), 0)
                        img1_patch = img1[y:y+patch_y]
                        img2_patch = img2[y:y+patch_y]
                    # patches to gpu
                    img1_patch_gpu = torch.from_numpy(img1_patch).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
                    img2_patch_gpu = torch.from_numpy(img2_patch).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
                    #print(img1_patch_gpu.shape, img2_patch_gpu.shape)
                    padder = InputPadder(img1_patch_gpu.shape)
                    img1_patch_gpu, img2_patch_gpu = padder.pad(img1_patch_gpu, img2_patch_gpu)

                    #print(img1_patch_gpu.shape, img2_patch_gpu.shape)
                    flow_up_sub = self.model(img1_patch_gpu, img2_patch_gpu)
                    flow_up_unp_sub = padder.unpad(flow_up_sub[0])
                    flow_np = flow_up_unp_sub.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                    # Add flow to result and update weights for averaging
                    result[y:min(y + patch_y, img1.shape[0]), :, :] += flow_np[:min(patch_y, img1.shape[0] - y), :, :]
                    weights[y:min(y + patch_y, img1.shape[0]), :, :] += 1

                    del img1_patch_gpu, img2_patch_gpu#, model
                    torch.cuda.empty_cache()
            
            result = result / np.maximum(weights, 1e-6)  # Avoid division by zero

        elif orientation == 'horizontal':
            patch_y = img1.shape[0]
            patch_x = maxpix // img1.shape[0]
            overlap = (int)(patch_x * 0.1)

            result = np.zeros((img1.shape[0], img1.shape[1], 2), dtype=np.float32)
            weights = np.zeros((img1.shape[0], img1.shape[1], 1), dtype=np.float32)
            with torch.no_grad():
                if not self.silent:
                    print(f"Processing {img1.shape[1] // (patch_x - overlap)} flow patches {patch_y} x {patch_x}")
                for x in range(0, img1.shape[1], patch_x - overlap):
                    #print(f"Processing path {x // (patch_x - overlap)} of {img1.shape[1] // (patch_x - overlap)}")

                    #model = raft_large(weights=self.weights)
                    #model = model.eval()
                    #model = model.to(self.device)

                    # extract patches, respect image boundaries
                    img1_patch = img1[:, x:x+patch_x]
                    img2_patch = img2[:, x:x+patch_x]
                    # if patch is too small, move it left
                    if img1_patch.shape[1] < patch_x:
                        x = max(x - (patch_x - img1_patch.shape[1]), 0)
                        img1_patch = img1[:, x:x+patch_x]
                        img2_patch = img2[:, x:x+patch_x]
                    # patches to gpu
                    img1_patch_gpu = torch.from_numpy(img1_patch).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
                    img2_patch_gpu = torch.from_numpy(img2_patch).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0

                    padder = InputPadder(img1_patch_gpu.shape)
                    img1_patch_gpu, img2_patch_gpu = padder.pad(img1_patch_gpu, img2_patch_gpu)

                    flow_up_sub = self.model(img1_patch_gpu, img2_patch_gpu)
                    flow_up_unp_sub = padder.unpad(flow_up_sub[0])
                    flow_np = flow_up_unp_sub.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                    # Add flow to result and update weights for averaging
                    result[:, x:min(x + patch_x, img1.shape[1]), :] += flow_np[:, :min(patch_x, img1.shape[1] - x), :]
                    weights[:, x:min(x + patch_x, img1.shape[1]), :] += 1
                    
                    del img1_patch_gpu, img2_patch_gpu#, model
                    torch.cuda.empty_cache()
            
            result = result / np.maximum(weights, 1e-6)  # Avoid division by zero

        # Final averaging to handle overlaps
        return result

    def compute_optical_flow(self, img1_overlap, img2_overlap, orig_size, subsample=1.0):
        #if subsample < 0:
        #    # auto compute max possible subsample
        #    maxpix = 1200 * 1200
        #    subsample = np.sqrt((img1_overlap.shape[2] * img1_overlap.shape[3]) / maxpix)
        #    if not self.silent:
        #        print(f"Auto computed subsample: {subsample}")

        #if subsample != 1.0:
        #    img1_overlap_sub = Fnn.interpolate(img1_overlap, scale_factor=1.0/subsample, mode='bilinear', align_corners=False)
        #    img2_overlap_sub = Fnn.interpolate(img2_overlap, scale_factor=1.0/subsample, mode='bilinear', align_corners=False)
        #else:
        img1_overlap_sub, img2_overlap_sub = img1_overlap, img2_overlap

        # Apply padding to subsampled images
        padder = InputPadder(img1_overlap_sub.shape)
        img1_overlap_sub, img2_overlap_sub = padder.pad(img1_overlap_sub, img2_overlap_sub)

        #print(img1_overlap_sub.shape, img2_overlap_sub.shape)
        
        # Compute optical flow on the subsampled images
        with torch.no_grad():
            flow_up_sub = self.model(img1_overlap_sub, img2_overlap_sub)
            # Remove padding from the subsampled flow
            flow_up_unp_sub = padder.unpad(flow_up_sub[0])

            # Upsample the flow back to the original resolution
            # print(flow_up_unp_sub.shape, img1_overlap.shape)
            flow_up_unp = Fnn.interpolate(flow_up_unp_sub, size=(orig_size.shape[0], orig_size.shape[1]), mode='nearest')
            
            if subsample != 1.0:
                flow_up_unp = flow_up_unp * (subsample)
            
        return flow_up_unp.squeeze(0).cpu().numpy()

    def process_image(self, image_list1, image_list2, size):
        """
        Processes the images to extract the optical flow.
        """

        img1_batch = torch.stack(image_list1)
        img2_batch = torch.stack(image_list2)
        # convert images to pytorch tensors

        img1_batch = F.resize(img1_batch, size, antialias=False)
        img2_batch = F.resize(img2_batch, size, antialias=False)
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)
        # preprocess the images

        list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))

        flow_tensors = [flow for flow in list_of_flows[-1]]
        flow_batch = torch.stack(flow_tensors)

        if self.debug:
            flow_imgs = flow_to_image(flow_batch)
            img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
            grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
            self.plot(grid)

        return flow_batch
    
    def plot(self, imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
        plt.show()

    def process_images(self, image_list1, image_list2, size):
        """
        Processes the images to extract the optical flow.
        """

        img1_batch = torch.stack(image_list1)
        img2_batch = torch.stack(image_list2)
        # convert images to pytorch tensors

        img1_batch = F.resize(img1_batch, size, antialias=False)
        img2_batch = F.resize(img2_batch, size, antialias=False)
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)
        # preprocess the images, todo: patch size

        list_of_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))

        flow_tensors = [flow for flow in list_of_flows[-1]]
        flow_batch = torch.stack(flow_tensors)

        if self.debug:
            flow_imgs = flow_to_image(flow_batch)
            img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
            grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
            self.plot(grid)

        return flow_batch
    
    def plot(self, imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
        plt.show()