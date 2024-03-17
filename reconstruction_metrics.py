import torch
import torch.nn.functional as F
import numpy as np
import math

def ssim(img1, img2, window_size=11, window_sigma=1.5, data_range=1.0):
    """
    Calculate the structural similarity index (SSIM) between two images.

    Args:
        img1 (torch.Tensor): First image.
        img2 (torch.Tensor): Second image.
        window_size (int): Size of the sliding window. Default is 11.
        window_sigma (float): Standard deviation of the sliding window. Default is 1.5.
        data_range (float): Range of the input data. Default is 1.0.

    Returns:
        float: SSIM value between the two images.
    """
    from skimage.metrics import structural_similarity as ssim_skimage

    img1 = img1.permute(0, 2, 3, 1).cpu().numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().numpy()

    ssim_value = 0.0
    for i in range(len(img1)):
        ssim_value += ssim_skimage(img1[i], img2[i], win_size=window_size, sigma=window_sigma, data_range=data_range, multichannel=True)

    ssim_value /= len(img1)

    return ssim_value

def psnr(img1, img2, data_range=1.0):
    """
    Calculate the peak signal-to-noise ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor): First image.
        img2 (torch.Tensor): Second image.
        data_range (float): Range of the input data. Default is 1.0.

    Returns:
        float: PSNR value between the two images.
    """
    mse = F.mse_loss(img1, img2)
    psnr_value = 20 * math.log10(data_range / math.sqrt(mse.item()))
    return psnr_value


def calculate_METRICS(model, dataloader, device='cuda'):
    """
    Calculate the reconstruction loss, RMSE, SSIM, and PSNR for an autoencoder model using a DataLoader.

    Args:
        model (torch.nn.Module): The autoencoder model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to use for computation. Default is 'cuda'.

    Returns:
        float: The average reconstruction loss.
        float: The average RMSE.
        float: The average SSIM.
        float: The average PSNR.
    """
    # Set the model to evaluation mode
    model.eval()

    total_loss_mse = 0.0
    total_loss_mae = 0.0
    total_rmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    num_samples = 0

    with torch.no_grad():
        torch.manual_seed(2809)

        for batch in dataloader:
            input_image, output_image = batch["input_image"].float().to(device), batch["output_image"].float().to(device)

            # Forward pass to obtain the reconstructed output
            reconstructed_data = model(input_image)

            # Calculate the mean squared error (MSE) loss for this batch
            batch_loss_mse = F.mse_loss(reconstructed_data, output_image)
            total_loss_mse += batch_loss_mse.item()

            # Calculate the root mean squared error (RMSE) for this batch
            batch_rmse = math.sqrt(batch_loss_mse.item())
            total_rmse += batch_rmse

            # Calculate the structural similarity index (SSIM) for this batch
            batch_ssim = ssim(reconstructed_data, output_image)
            total_ssim += batch_ssim

            # Calculate the peak signal-to-noise ratio (PSNR) for this batch
            batch_psnr = psnr(reconstructed_data, output_image)
            total_psnr += batch_psnr

            batch_loss_mae = F.l1_loss(reconstructed_data, output_image)
            total_loss_mae += batch_loss_mae.item()

            num_samples += len(batch)

    # Calculate the average loss, RMSE, SSIM, and PSNR
    average_loss_mse = total_loss_mse / len(dataloader)
    average_loss_mae = total_loss_mae / len(dataloader)
    average_rmse = total_rmse / len(dataloader)
    average_ssim = total_ssim / len(dataloader)
    average_psnr = total_psnr / len(dataloader)

    return average_loss_mse, average_loss_mae, average_rmse, average_ssim, average_psnr
