import json

class Config:

    # Setting default values for the config variables
    epochs = 1000
    layers = [784, 256, 10]
    output_period = 5

    @staticmethod
    def import_from_file(file_name):
        try:
            with open(file_name) as json_data_file:
                data = json.load(json_data_file)
        except:
            print("Bad JSON file. Failed to read")
            return

        for key, value in data.items():
            setattr(Config, key, value)


