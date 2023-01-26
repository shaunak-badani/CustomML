from micrograd.nn import MLP
from micrograd.engine import Value
from data_loaders.mnist import MNISTData
from config import Config

import argparse
import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help = "optional config file")
args = parser.parse_args()

if args.config:
    Config.import_from_file(args.config)
    

mnist_data_object = MNISTData()
model = MLP(Config.layers)

if hasattr(Config, 'state_dict_path'):
    model = MLP.load_to_model(Config.state_dict_path, Config.layers)
data, target = mnist_data_object.load_flattened_batch()
history = model.train(Value(data), target, loss_fn = 'cross_entropy', epochs = Config.epochs, output_period = 5)
print(history)
save_path = "../models"
model_name = "model"
if hasattr(Config, 'run_name'):
    model_name = Config.run_name
save_path = os.path.join(save_path, model_name)

model.save_to_torch_model(save_path)

# m2 = MLP.load_to_model()
# print(m2)