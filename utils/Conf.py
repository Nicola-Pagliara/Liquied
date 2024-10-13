import yaml


class Config:
    def __init__(self, data_par, train_par, model_par):
        self.data_par = data_par
        self.train_par = train_par
        self.model_par = model_par

    @classmethod
    def from_yaml(cls):
        with open('Configs/configuration.yaml', 'r') as file:
            params = yaml.safe_load(file.read())
            file.close()
        return cls(params.data, params.train, params.model)
