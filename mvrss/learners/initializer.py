"""Initializer class to prepare training"""
import json
from torch.utils.data import DataLoader

from utils.paths import Paths
from loaders.dataset import Carrada
from loaders.dataloaders import SequenceCarradaDataset


class Initializer:
    """Class to prepare training model

    PARAMETERS
    ----------
    cfg: dict
        Configuration file used for train/test
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = Paths().get()

    def _get_data(self):
        data = Carrada()
        train = data.get('Train')
        val = data.get('Validation')
        test = data.get('Test')
        return [train, val, test]

    def _get_datasets(self):
        data = self._get_data()

        train_num_sequences = self.cfg.get('train_num_sequences', None)
        val_num_sequences = self.cfg.get('val_num_sequences', None)
        test_num_sequences = self.cfg.get('test_num_sequences', None)

        random_select = self.cfg.get('random_select_sequences', False)
        seed = self.cfg.get('sequence_selection_seed', 42)

        trainset = SequenceCarradaDataset(
            data[0],
            num_sequences=train_num_sequences,
            random_select=random_select,
            seed=seed
        )
        valset = SequenceCarradaDataset(
            data[1],
            num_sequences=val_num_sequences,
            random_select=random_select,
            seed=seed
        )
        testset = SequenceCarradaDataset(
            data[2],
            num_sequences=test_num_sequences,
            random_select=random_select,
            seed=seed
        )

        return [trainset, valset, testset]

    def _get_dataloaders(self):
        trainset, valset, testset = self._get_datasets()
        trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        return [trainloader, valloader, testloader]

    def _structure_data(self):
        data = dict()
        dataloaders = self._get_dataloaders()

        ############################# name change stuff

        
        name_exp = (self.cfg['model'])
        self.cfg['name_exp'] = name_exp
        folder_path = self.paths['logs'] / self.cfg['dataset'] / self.cfg['model'] / name_exp

        temp_folder_name = folder_path.name + '_' + str(self.cfg['version'])
        temp_folder_path = folder_path.parent / temp_folder_name
        while temp_folder_path.exists():
            self.cfg['version'] += 1
            temp_folder_name = folder_path.name + '_' + str(self.cfg['version'])
            temp_folder_path = folder_path.parent / temp_folder_name
        folder_path = temp_folder_path

        self.paths['results'] = folder_path / 'results'
        self.paths['writer'] = folder_path / 'boards'
        self.paths['results'].mkdir(parents=True, exist_ok=True)
        self.paths['writer'].mkdir(parents=True, exist_ok=True)
        
        ##################### change name


        config_path = folder_path / 'config.json'
        with open(config_path, 'w') as fp:
            json.dump(self.cfg, fp)

        data['cfg'] = self.cfg
        data['paths'] = self.paths
        data['dataloaders'] = dataloaders
        return data

    def get_data(self):
        """Return parameters of the training"""
        return self._structure_data()