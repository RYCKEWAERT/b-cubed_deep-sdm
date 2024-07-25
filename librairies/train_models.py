import os
import wandb
from librairies.mlp import train_custom_model_realdataset,train_classification_model
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from librairies.utils import set_seed
import verde as vd

def train_and_eval_models_from_elith(args):

    trainPOfile = args.dirdata + args.region + "train_po.csv"
    test_PAfile = args.dirdata + args.region + "test_pa.csv"
    testenv_PAfile = args.dirdata + args.region + "test_env.csv"
    PO = pd.read_csv(trainPOfile)
    PA = pd.read_csv(test_PAfile)
    PA_env_df = pd.read_csv(testenv_PAfile)
    
    variables = ['bcc', 'calc', 'ccc', 'ddeg', 'nutri', 'pday', 'precyy', 'sfroyy', 'slope', 'sradyy', 'swb', 'tavecc', 'topo']
    
    X_train = torch.tensor(PO[variables].to_numpy(), dtype=torch.float32)
    X_test =  torch.tensor(PA_env_df[variables].to_numpy(), dtype=torch.float32)

    y_test_pa = PA.iloc[:,4::]
    y_train_po = np.zeros((PO.shape[0], y_test_pa.shape[1]))
    icolumn = y_test_pa.columns
    for i in range(PO.shape[0]):
        spid_value = PO.loc[i, 'spid'] 
        idx = np.where(icolumn == spid_value)[0]
        if idx.size > 0:
            y_train_po[i,idx] = 1
        
        
    y_test = torch.tensor(y_test_pa.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train_po, dtype=torch.float32)
    
    scaler = StandardScaler()
    X_train_scaled= torch.tensor(
        scaler.fit_transform(X_train), dtype=torch.float32
    )

    X_test_scaled = torch.tensor(
        scaler.transform(X_test), dtype=torch.float32
    )


    args.learning_rate = args.conditions["default"]["learning_rate"]
    args.epoch = args.conditions["default"]["epoch"]
    args.hidden_size = args.conditions["default"]["hidden_size"]


    for idx_seed in range(args.repeat_seed):
        wandbname =str(args.list_of_seed[idx_seed])
        wandb.init(
            name = wandbname,
            mode=args.mode_wandb,
            project="disentangling-method",
            config={
                "seed": args.list_of_seed[idx_seed]
            }
        )
        
        # Set the random seed for reproducibility
        set_seed(args.list_of_seed[idx_seed])
        results = train_custom_model_realdataset(
            X_train_scaled,
            y_train,
            args,
            hidden_size=args.hidden_size,
            device="cuda"
        )

        model = results["model"]
        filename_model = f"{args.list_of_seed[idx_seed]}"
        full_path = f"{args.outputdir}/models/{filename_model}.pth"

        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'columns': variables
        }, full_path)   
        
        
        
        with torch.no_grad():
            predictions = model(X_test_scaled)

        predictions = predictions 
        predictions = torch.clamp(predictions,
                                    min=-np.inf,
                                    max=88.7)
        predictions = predictions.exp().detach().cpu().numpy()
        predictions[np.isinf(predictions)] = np.finfo(np.float32).max
        auc = roc_auc_score(y_test, predictions)
        print(f"AUC: {auc}")
        wandb.finish()
        

def train_and_eval_models_belgium(args,tensor,df):
    spid = df['species'].unique()
    y_target = np.zeros((df.shape[0], len(spid)))
    icolumn = df.columns
    for i in range(df.shape[0]):
        species_value = df.loc[i, 'species']
        idx = np.where(spid == species_value)[0]
        if idx.size > 0:
            y_target[i, idx] = 1
    idle = np.arange(df.shape[0])
    # Split data into train, validation and test sets using verede
    idle_cal, idle_tmp = vd.train_test_split(
        [df['decimallatitude'],df['decimallongitude']], idle, random_state=42, test_size=0.30
    )
    X_train = torch.tensor(tensor[idle_cal[1]], dtype=torch.float32)
    df_tmp = df.iloc[idle_tmp[1]]

    idle_val, idle_test = vd.train_test_split(
        [df_tmp['decimallatitude'],df_tmp['decimallongitude']], idle_tmp[1], random_state=42, test_size=0.50
    )
    
    X_val = torch.tensor(tensor[idle_val[1]], dtype=torch.float32)
    X_test = torch.tensor(tensor[idle_test[1]], dtype=torch.float32)
    y_train = torch.tensor(y_target[idle_cal[1]], dtype=torch.float32)
    y_val = torch.tensor(y_target[idle_val[1]], dtype=torch.float32)
    y_test = torch.tensor(y_target[idle_test[1]], dtype=torch.float32)
    
    # Get TRUE LABEL 
    
    scaler = StandardScaler()
    X_train_scaled= torch.tensor(
        scaler.fit_transform(X_train), dtype=torch.float32
    )
    X_val_scaled = torch.tensor(
        scaler.transform(X_val), dtype=torch.float32
    )
    X_test_scaled = torch.tensor(
        scaler.transform(X_test), dtype=torch.float32
    )


    args.learning_rate = args.conditions["default"]["learning_rate"]
    args.epoch = args.conditions["default"]["epoch"]
    args.hidden_size = args.conditions["default"]["hidden_size"]


    for idx_seed in range(args.repeat_seed):
        wandbname =str(args.list_of_seed[idx_seed])
        wandb.init(
            name = wandbname,
            mode=args.mode_wandb,
            project="B-cubed-Belgium",
            config={
                "seed": args.list_of_seed[idx_seed]
            }
        )
        
        # Set the random seed for reproducibility
        set_seed(args.list_of_seed[idx_seed])
        results = train_classification_model(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            args,
            hidden_size=args.hidden_size,
            device="cuda"
        )

        model = results["model"]
        with torch.no_grad():
            predictions = model(X_test_scaled)
            pred_PA = predictions.cpu().numpy()
    
            # Convertir les logits en prédictions binaires
            pred_labels = np.argmax(pred_PA, axis=1)
            true_labels = np.argmax(y_test.cpu().numpy(), axis=1)
            
            # Calculer les métriques
            precision = precision_score(true_labels, pred_labels, average='weighted')
            recall = recall_score(true_labels, pred_labels, average='weighted')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1: {f1}")
        wandb.finish()


def make_results_directory(args):
    if not os.path.exists(f"{args.outputdir}"):
        os.mkdir(f"{args.outputdir}")
    if not os.path.exists(f"{args.outputdir}/models"):
        os.mkdir(f"{args.outputdir}/models")
