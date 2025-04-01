import argparse
import torch
import datetime
import json
import yaml
import os

from src.main_model_table import TabCSDI
from src.utils_table import evaluate
from testing_airquality_real import get_dataloader

parser = argparse.ArgumentParser(description="TabCSDI")
parser.add_argument("--config", type=str, default="airquality.yaml")
parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

# Create folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/airquality_2_train_testing_fold" + str(args.nfold) + "_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
test_loader = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = TabCSDI(config, args.device).to(args.device)

model.load_state_dict(torch.load("./save/" +"airquality_2_fold5_20250219_203046"+ "/model.pth"))
print("---------------Start testing---------------")
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
