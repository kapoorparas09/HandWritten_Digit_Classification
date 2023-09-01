from src.utils.common import read_config
from src.utils.data_managment import get_data
import argparse

def training(config_path):
    config = read_config(config_path)

    validation_datasize = config["params"]["validation_data_size"]
    (X_train, y_train), (X_test, y_test),(X_valid,y_valid)= get_data(validation_datasize)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)