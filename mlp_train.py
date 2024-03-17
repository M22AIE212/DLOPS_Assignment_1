## Setting up Tensorboard
from torch.utils.tensorboard import SummaryWriter


import torch
from torch.utils.data import Dataset ,DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import seaborn as sns

from autoencoder import Autoencoder
from dataset_hymenoptera import HYMENOPTERADataset
from mlp import MLPClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Parameters")
    parser.add_argument('--btl_nck_dim', type=int, default=400, help='Bottleneck dimension')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training')
    parser.add_argument('--input_dim', type=int, default=512*28*28, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for MLP')
    parser.add_argument('--output_dim', type=int, default=2, help='Number of output classes')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_hidden_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--best_autoencoder_model_path', type=str, default=30, help='Best Autoencoder Path')
    parser.add_argument('--dir', type=str, default=os.getcwd(), help='Loss Directory path')
    parser.add_argument('--task', type=str, default='lab_to_rgb', help='Task description, "lab_to_rgb" or "rgb_to_hflip" or "rgb_to_neg"')
    parser.add_argument('--mlp_model_dir', type=str, default=os.getcwd(), help='Model path')
    args = parser.parse_args()

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'/content/drive/MyDrive/Assignments/DLOPS/Assignment1/codes/runs_{args.task}/mlp_classifier_experiment')

    # Data Directory Paths
    train_data_dir = "./hymenoptera_data/hymenoptera_data/train"
    test_data_dir = "./hymenoptera_data/hymenoptera_data/val"

    # Loss Directory Paths
    train_loss_dir_path = args.dir
    val_loss_dir_path = args.dir

    # Parameters
    best_vloss = 1_000_000
    btl_nck_dim = args.btl_nck_dim
    device = args.device
    input_dim = args.input_dim  # output dimension of encoder
    hidden_dim = args.hidden_dim  # hidden dimension for MLP
    output_dim = args.output_dim  # number of output classes
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_hidden_layers = args.num_hidden_layers
    print(f"Number of hidden layers in MLP {num_hidden_layers}")
    epochs = args.epochs
    model_path = args.best_autoencoder_model_path
    task = args.task
    mlp_model_dir = args.mlp_model_dir

  
    target_classes = ["ants","bees"]
    ## Creating Datasets
    train_dataset = HYMENOPTERADataset(train_data_dir)
    test_dataset = HYMENOPTERADataset(test_data_dir)

    ## Creating Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ## Loading Autoencoder
    autoencoder = Autoencoder(btl_nck_dim=btl_nck_dim)
    autoencoder.load_state_dict(torch.load(model_path, map_location='cuda'))
    autoencoder = autoencoder.to(device)


    # Initialize downstream task classifier
    classifier = MLPClassifier(input_dim, hidden_dim, output_dim,num_hidden_layers).to(device)

    # Freeze the encoder's weights
    for param in autoencoder.parameters():
        param.requires_grad = False

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    train_loss = []
    val_loss = []

    # Training loop
    for epoch in range(epochs):
        running_train_loss = 0
        for data in tqdm(train_dataloader):
            inputs, labels,_ = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # Feature Extractor : Extract features using the encoder of the pretrained autoencoder
            encoder_output = autoencoder.encoder(inputs)

            # Flatten the encoder output if needed
            # encoder_output = encoder_output.view(encoder_output.size(0), -1)

            # Forward pass
            outputs = classifier(encoder_output)
            loss = criterion(outputs, labels)
            running_train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss =  (running_train_loss / len(train_dataloader.dataset))
        writer.add_scalar("Training Loss MLP Classifier" , avg_loss, epoch + 1)
        running_vloss = 0.0

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        classifier.eval()
        total = 0
        correct = 0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for vdata in tqdm(val_dataloader):

                vinputs, vlabels,_ = vdata
                vinputs = vinputs.cuda()
                vlabels = vlabels.cuda()
                encoder_output = autoencoder.encoder(vinputs)
                voutputs = classifier(encoder_output)
                _, predicted = torch.max(voutputs.data, 1)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
                total += vlabels.size(0)
                correct += (predicted == vlabels).sum().item()

        accuracy = 100 *(correct/total)
        writer.add_scalar("Validation Accuracy MLP Classifier" , accuracy, epoch + 1)

        avg_vloss = (running_vloss / len(val_dataloader.dataset))
        writer.add_scalar("Validation Loss MLP Classifier" , avg_vloss, epoch + 1)
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{mlp_model_dir}/task_{task}_classifier.pth'
            torch.save(classifier.state_dict(), model_path)

        train_loss.append((epoch,avg_loss))
        val_loss.append((epoch,avg_vloss))
        print()
        #print('Epochs {} LOSS train : {} , valid {} , LR : {}'.format(epoch,avg_loss, avg_vloss,optimizer.param_groups[0]['lr']))
        print('Epochs {} LOSS value : {} , LR : {}'.format(epoch,avg_loss,optimizer.param_groups[0]['lr']))
        print()
    train_loss_df = pd.DataFrame(train_loss,columns = ['epoch','loss'])
    val_loss_df = pd.DataFrame(val_loss,columns = ['epoch','loss'])
    train_loss_df.to_csv(train_loss_dir_path + f"/task_{task}_hymenopter_train_loss_epochs_{epochs}_num_hidden_layers_{num_hidden_layers}.csv")
    val_loss_df.to_csv(val_loss_dir_path + f"/task_{task}_hymenopter_val_loss_epochs_{epochs}_num_hidden_layers_{num_hidden_layers}.csv")

    ## Prepare confusion matrix and AUC-ROC curve to evaluate the model performance
    # Evaluate the model on the test set
    classifier.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        y_pred_t = []
        for inputs, labels,_ in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            encoder_output = autoencoder.encoder(inputs)
            outputs = classifier(encoder_output)
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.cpu().numpy().tolist()
            y_pred += predicted.cpu().numpy().tolist()
            y_pred_t.append(outputs)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print the confusion matrix
    sns.set(font_scale=0.8)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=target_classes, yticklabels=target_classes)

    # Add labels and title
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate the AUC-ROC score
    y_pred_all = torch.cat(y_pred_t)
    print(y_pred_all.shape)
    y_score = torch.softmax(torch.cat(y_pred_t),dim = 1).cpu().numpy()
    print(y_score.shape)
    auc_roc_score = roc_auc_score(y_true, y_score[:,1])

    # Print the AUC-ROC score
    print("AUC-ROC score: {:.4f}".format(auc_roc_score))

    import sklearn.metrics as metrics
    # Calculate the AUC-ROC curve
    fpr, tpr, threshold = roc_curve(y_true, y_score[:,1])
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    writer.close()
