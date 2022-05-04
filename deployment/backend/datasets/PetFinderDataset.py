import pandas as pd
from pathlib import Path
from .Dataset import Dataset

DATASET_PATH = Path('datasets/csv').absolute()


class PetFinderDataset(Dataset):

    def __init__(self, path=DATASET_PATH):
        self.path = path
        self.original_dataframe = pd.read_csv(f"{self.path}/train.csv")
        self.dataframe = self.preprocessing(self.original_dataframe)

        super().__init__(x=self.dataframe.drop(columns=['AdoptionSpeed']), y=self.dataframe['AdoptionSpeed'])

    def preprocessing(self, _dataset):
        _dataset['PhotoAmt'] = _dataset['PhotoAmt'].astype(int)
        _dataset = _dataset.replace(self.get_feature_map())
        _dataset = _dataset.replace({'Breed1': {0: "Not Specified"}})
        _dataset = _dataset.replace({'Breed2': {0: "Not Specified"}})
        _dataset = _dataset.replace({'Breed3': {0: "Not Specified"}})
        _dataset = _dataset.replace({'Color1': {0: "Not Specified"}})
        _dataset = _dataset.replace({'Color2': {0: "Not Specified"}})
        _dataset = _dataset.replace({'Color3': {0: "Not Specified"}})
        _dataset = _dataset.drop(['RescuerID'], axis=1)
        _dataset = _dataset.set_index('PetID')
        return _dataset

    def get_feature_map(self):
        breed_df = pd.read_csv(f'{self.path}/breed_labels.csv')
        breed_map = breed_df.set_index('BreedID')['BreedName'].to_dict()

        color_df = pd.read_csv(f'{self.path}/color_labels.csv')
        color_map = color_df.set_index('ColorID').to_dict()['ColorName']

        state_df = pd.read_csv(f'{self.path}/state_labels.csv')
        state_map = state_df.set_index('StateID')['StateName'].to_dict()

        type_map = {1: 'Dog', 2: 'Cat'}
        gender_map = {1: 'Male', 2: 'Female', 3: 'Mixed'}
        maturity_size_map = {1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large', 0: 'Not Specified'}
        fur_length_map = {1: 'Short', 2: 'Medium', 3: 'Long', 0: 'Not Specified'}
        boolean_map = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
        health_map = {1: 'Healthy', 2: 'Minor Injury', 3: 'Serious Injury', 4: 'Not Specified'}

        return {
            'Type': type_map,
            'Breed1': breed_map,
            'Breed2': breed_map,
            'Gender': gender_map,
            'Color1': color_map,
            'Color2': color_map,
            'Color3': color_map,
            'MaturitySize': maturity_size_map,
            'FurLength': fur_length_map,
            'Vaccinated': boolean_map,
            'Dewormed': boolean_map,
            'Sterilized': boolean_map,
            'Health': health_map,
            'State': state_map,
        }
