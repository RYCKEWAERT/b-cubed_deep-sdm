########################################################################################
# Launcher to run a subset of the Elith et al. 2020 dataset, 
# The data is composed of SWI region
# The model is calibrated on presence-only data using a MLP to have species intensities
# The evaluation is performed with AUC on the test set of Presence Absence data

from librairies.train_models import train_and_eval_models_from_elith
from librairies.utils import set_seed, get_random_seed_list, disable_warnings
import os



class ConfigArgs:
    def __init__(self, config_file=None):
        if config_file:
            self.load_configuration(config_file)
        else:
            self.load_default_configuration()

    def load_default_configuration(self):
        self.outputdir = "results_Elith"
        self.dirdata = "./data/Elith/"
        self.global_seed = 42  # the answer to life the universe and everything

        self.conditions = {
            "default": {
                "learning_rate": 0.01,
                "epoch": 10,
                "hidden_size": 250
            },
        }
        self.mode_wandb = 'offline' # 'online' or 'offline'
        os.environ['WANDB_SILENT'] = "true"
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        set_seed(self.global_seed)
        self.list_of_seed = get_random_seed_list(1000)
        self.repeat_seed = 1
        self.region = "SWI"
 
def run(args):
    disable_warnings()  # Ignore all warnings during the code's execution
    set_seed(args.global_seed)
    train_and_eval_models_from_elith(args)

if __name__ == "__main__":
    args = ConfigArgs()
    run(args)
    os.system('notify-send "Script ended" "And correctly!"')
