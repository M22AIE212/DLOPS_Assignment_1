import torch
import numpy as np

def train_one_epoch(epoch_index,train_dataloader,model,loss_fn,optimizer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    torch.manual_seed(2809)
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        input_image, output_image =  data["input_image"].float().cuda(),data["output_image"].float().cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        recon_outputs = model(input_image)

        # Compute the loss and its gradients
        loss = loss_fn(recon_outputs, output_image)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        # scheduler.step()

        # Gather data and report
        running_loss += loss.item()
    epoch_loss = (running_loss / len(train_dataloader))

    return epoch_loss , model
