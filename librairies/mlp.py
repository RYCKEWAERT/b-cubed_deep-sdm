import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()

        self.fc1_lambda = nn.Linear(input_size, hidden_size)
        self.fc2_lambda = nn.Linear(hidden_size, hidden_size)
        self.fc3_lambda = nn.Linear(hidden_size, output_size) 
        

    def forward(self, xinput):
        x = self.fc1_lambda(xinput).relu()
        x = self.fc2_lambda(x).relu()
        x = self.fc3_lambda(x)

        return x

    
def save_mlp_model(args, model):
    """
    Save the MLP model to a PyTorch model file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): MLP model.
    """
    mlp_filepath = (
        args.outputdir + "model/MLP.pth"
    )  # Define the file path to save the MLP model file
    torch.save(model, mlp_filepath)  # Save the MLP model to a PyTorch model file


def load_mlp_model(args, model):
    """
    Load the pre-trained MLP model.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): The MLP model object.

    Returns:
        models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel: The loaded pre-trained MLP model.
    """
    mlp_filepath = (
        args.outputdir + "model/MLP.pth"
    )  # Filepath of the pre-trained MLP model
    model.load_state_dict(
        torch.load(mlp_filepath)
    )  # Load the weights of the pre-trained model
    return model


def make_predictions(model, X_tensor):
    """
    Make predictions using the given PyTorch model and input tensor.

    Parameters:
        model (torch.nn.Module): The PyTorch model.
        X_tensor (torch.Tensor): The input tensor for making predictions.

    Returns:
        torch.Tensor: The predictions.
    """
    model.eval()
    model = model.to("cpu")
    X_tensor = X_tensor.to("cpu")

    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions

class loss_Poisson(nn.Module):
    def __init__(self):
        super(loss_Poisson, self).__init__()

    def forward(self, input, target):
        lbd = input
        loss_lambda = F.poisson_nll_loss(lbd, target, log_input=True)

        return loss_lambda
    
    
def train_custom_model_realdataset(
    X_tensor,
    y_tensor, 
    args,
    hidden_size=250,
    device="cuda"
):

    X_tens_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(X_tens_data, batch_size=250, shuffle=True)
    
    criterion = loss_Poisson().to(device)

        
    input_size = X_tensor.shape[1]
    output_size = y_tensor.shape[1]
    
    model = SimpleMLP(input_size, hidden_size, output_size) 
    model = model.to(device)

    optimizer = optim.Adam(
        [
            {"params": model.parameters(), "lr": args.learning_rate},
        ],
    )

    num_epochs = args.epoch
    loss_by_batch = []
    best_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss_train = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
        
            loss = criterion(outputs, batch_y)
        
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item() 
            
           
        # Losses monitoring
        loss_by_batch.append(total_loss_train / len(train_loader))
        
        wandb.log({"epoch": epoch,
                    "total_loss": total_loss_train / len(train_loader)})
        
        if total_loss_train / len(train_loader) < best_loss:
            best_loss = total_loss_train / len(train_loader)
            best_model = copy.deepcopy(model.state_dict())
                
    model.load_state_dict(best_model)
    predictions = make_predictions(model, X_tensor)
    result = {
        "predictions": predictions,
        "model": model,
    }
    return result



def train_classification_model(
    X_tensor,
    y_tensor, 
    X_tensor_val,
    y_tensor_val, 
    args,
    hidden_size=250,
    device="cuda"
):

    X_tens_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(X_tens_data, batch_size=250, shuffle=True)
    
    criterion = nn.CrossEntropyLoss().to(device)

        
    input_size = X_tensor.shape[1]
    output_size = y_tensor.shape[1]
    
    model = SimpleMLP(input_size, hidden_size, output_size) 
    model = model.to(device)

    optimizer = optim.Adam(
        [
            {"params": model.parameters(), "lr": args.learning_rate},
        ],
    )

    num_epochs = args.epoch
    loss_by_batch = []
    best_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss_train = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
        
            loss = criterion(outputs, batch_y)
        
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item() 
            
           
        # Losses monitoring
        loss_by_batch.append(total_loss_train / len(train_loader))
        
        wandb.log({"epoch": epoch,
                    "total_loss": total_loss_train / len(train_loader)})
        
        if total_loss_train / len(train_loader) < best_loss:
            best_loss = total_loss_train / len(train_loader)
            best_model = copy.deepcopy(model.state_dict())
            
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs_PA = model(X_tensor_val.to(device))
            pred_PA = outputs_PA.cpu().numpy()
            
            # Convertir les logits en prédictions binaires
            pred_labels = np.argmax(pred_PA, axis=1)
            true_labels = np.argmax(y_tensor_val.cpu().numpy(), axis=1)
            
            # Calculer les métriques
            precision = precision_score(true_labels, pred_labels, average='weighted')
            recall = recall_score(true_labels, pred_labels, average='weighted')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            
            # Logguer les métriques
            wandb.log({"precision": precision,
                       "recall": recall,
                       "f1": f1})
        
    model.load_state_dict(best_model)
    predictions = make_predictions(model, X_tensor)
    result = {
        "predictions": predictions,
        "model": model,
    }
    return result
    