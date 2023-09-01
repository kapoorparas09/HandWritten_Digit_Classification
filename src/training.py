from src.utils.common import read_config
from src.utils.data_managment import get_data
from src.utils.models import create_model
import argparse

def training(config_path):
    config = read_config(config_path)

    validation_datasize = config["params"]["validation_data_size"]
    
    (X_train, y_train), (X_test, y_test),(X_valid,y_valid)= get_data(validation_datasize)
    
    LOSS_FUNCTION= config["params"]["loss_function"]
    OPTIMIZER= config["params"]["optimizer"]
    METRICS= config["params"]["metrics"]
    Num_classes= config["params"]["no_of_classes"]

    model = create_model(LOSS_FUNCTION,OPTIMIZER, METRICS, Num_classes )

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_data=VALIDATION_SET)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)