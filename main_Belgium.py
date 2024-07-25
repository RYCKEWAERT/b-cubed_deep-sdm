from librairies.train_models import train_and_eval_models_belgium
from librairies.utils import set_seed, get_random_seed_list, disable_warnings, import_belgium_data
import os

########################################################################################
# Launcher to run a BCUBED data from Belgium (2010), for species classification
# The data is composed of 19 raster files and a csv file with the species information
# The data is split into train, validation and test sets
# The model is a MLP with a CrossEntropy loss for classification

class ConfigArgs:
    def __init__(self, config_file=None):
        if config_file:
            self.load_configuration(config_file)
        else:
            self.load_default_configuration()

    def load_default_configuration(self):
        self.outputdir = "results_Belgium"
        self.dirdata = "./data/Belgium/"
        self.global_seed = 42  # the answer to life the universe and everything

        self.conditions = {
            "default": {
                "learning_rate": 0.01,
                "epoch": 20,
                "hidden_size": 250
            },
        }
        self.mode_wandb = 'offline' # 'online' or 'offline'
        os.environ['WANDB_SILENT'] = "true"
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        set_seed(self.global_seed)
        self.list_of_seed = get_random_seed_list(1000)
        self.repeat_seed = 1
        self.region = "Belgium"
 
def run(args):
    disable_warnings()  # Ignore all warnings during the code's execution
    set_seed(args.global_seed)
    tensor,df = import_belgium_data()
    train_and_eval_models_belgium(args, tensor, df)

if __name__ == "__main__":
    args = ConfigArgs()
    run(args)
    os.system('notify-send "Script ended" "And correctly!"')
