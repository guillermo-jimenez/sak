from typing import Tuple, Iterable, Union, List, Callable, Sized, Dict
import math
import torch
import torch.utils
import torch.utils.data
import numpy as np
import numpy.random
import wfdb
import os
import warnings
from scipy.interpolate import interp1d

from sak import class_selector
from sak.__ops import required
from sak.__ops import check_required


class UniformMultiDataset(torch.utils.data.Dataset):
    '''Composer that draws samples from multiple dataset classes. Only
    to be used with UniformMultiSampler'''

    def __init__(self, datasets: Iterable[torch.utils.data.Dataset], 
                 draws: Union[Iterable,int] = 1, 
                 weights: Union[Iterable,int] = 1,
                 return_weights: bool = False,
                 shuffle: bool = True):
        # Datasets
        if isinstance(datasets, Iterable):
            self.datasets = datasets
        else:
            raise ValueError("'datasets' must be an iterable")

        # Draws
        if isinstance(draws, Iterable):
            assert len(datasets) == len(draws)
            assert all([d > 0 for d in draws])
            self.draws = draws
        elif isinstance(draws, int):
            self.draws = [draws for _ in self.datasets]
        else:
            raise ValueError("'draws' must be an iterable or a single int")

        # Weights
        if isinstance(weights, Iterable):
            assert len(datasets) == len(weights)
            self.weights = weights
        elif isinstance(weights, int):
            self.weights = [weights for _ in self.datasets]
        else:
            raise ValueError("'weights' must be an iterable or a single int")
        
        # Compute number of generable samples & iterations
        self.iterations = math.ceil(max([len(d)/self.draws[i] for i,d in enumerate(self.datasets)]))
        self.len = self.iterations*sum(self.draws)
        self.indices = [np.arange(len(d)) for d in self.datasets]
        self.__return_weights = return_weights

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Retrieve indices
        i,g = index

        # Retrieve dataset
        dataset = self.datasets[g]
        
        # If the index exceeds the number of elements, use module
        i = i%len(dataset)
        
        # Define output
        out = {}
        out.update(dataset[i])
        if self.__return_weights: 
            out["weights"] = self.weights[g]
        
        return out

    
class UniformMultiSampler(torch.utils.data.Sampler):
    '''Random uniform sampler that oversamples the minority class according
    to the information in the dataset. To be used exclusively with the 
    UniformMultiDataset class'''

    def __init__(self, dataset: Sized) -> None:
        self.dataset = dataset
        self.groups = [len(d) for d in self.dataset.datasets]
        self.iterations = dataset.iterations
        self.len = dataset.len

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return self.len

    def __iter__(self):
        # Iterate over dataset
        ordering = np.random.permutation(np.arange(len(self.dataset.datasets)))
        index_dataset = np.concatenate([[i]*(self.dataset.draws[i]) for i in ordering])
        index_dataset = np.tile(index_dataset,self.iterations)

        # Iterate over windows
        index_windows = np.zeros_like(index_dataset)
        for i,draw in enumerate(self.dataset.draws):
            num_indices = draw*self.iterations
            index_temp = [np.random.permutation(start+np.arange(self.groups[i])) for start in range(0,num_indices,self.groups[i])]
            index_windows[index_dataset == i] = np.concatenate(index_temp)[:num_indices]
        
        # Instantiate generator
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        
        # Return iterable
        yield from zip(index_windows,index_dataset)

    def __len__(self):
        return self.len


class Dataset(torch.utils.data.Dataset):
    '''Generates data for PyTorch'''

    def __init__(self, X, y=None):
        '''Initialization'''
        
        # Store input
        try:
            self.X = torch.from_numpy(X)
        except:
            self.X = X
        try:
            self.y = torch.from_numpy(y)
        except:
            self.y = y

    def __len__(self):
        '''Denotes the number of elements in the dataset'''
        return len(self.X)

    def __getitem__(self, i: int):
        '''Generates one datapoint''' 
        return {"x": self.X[i], "y": self.y[i]}


class DatasetWFDB(torch.utils.data.Dataset):
    '''Generates data for PyTorch'''

    def __init__(self, samples: dict, beat_distribution: dict, root_dir: str, target_length: int):
        '''Initialization'''
        
        # Store input
        self.samples = samples
        self.root_dir = root_dir
        self.beat_distribution = beat_distribution
        self.target_length = target_length
        
        # Sanity check
        if np.sum([len(self.beat_distribution[s].keys()) == 0 for s in self.beat_distribution]) > 0:
            raise ValueError("Filter out empty symbols/databases/whatever")
        if np.any(np.array([np.sum([self.beat_distribution[s][fn] == 0 for fn in self.beat_distribution[s]]) for s in self.beat_distribution]) > 0):
            raise ValueError("Filter out empty symbols/databases/whatever")
            
        # Retrieve symbols and files
        self.symbols = np.array(list(self.beat_distribution.keys()))
        self.files = np.array([np.array(list(self.beat_distribution[self.symbols[i]].keys())) for i in range(len(self.symbols))])
        
        # Output shape tensor
        self.out_linspace = np.linspace(0,1,self.target_length)

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        pass

    def __getitem__(self, index: Tuple[np.ndarray,np.ndarray,np.ndarray]):
        '''Generates one datapoint''' 
        
        # Retrieve sub-indices
        (s, fn, b) = index
        
        # Get symbol
        symbol = self.symbols[s]
        
        # Get file name
        filecode = self.files[s][fn]
        
        # Get database, file, channel
        split_filecode = filecode.split('_')
        database = split_filecode[0]
        channel = int(split_filecode[-1])
        filename = '_'.join(split_filecode[1:-1])
        
        # Get onset and offset
        (on,off) = self.samples[symbol][database][filename][b,:]
        
        # Read file
        try:
            signal,_ = wfdb.rdsamp(os.path.join(self.root_dir,database,filename),sampfrom=on,sampto=off,channels=[channel])
        except ValueError:
            raise ValueError("(on > off) or (on < 0) or (off > N). Check annotations for file {}".format(os.path.join(self.root_dir,database,filename)))
        
        # Prepare to interpolate
        if on-off != self.target_length:
            signal = interp1d(np.linspace(0,1,off-on),signal.squeeze())(self.out_linspace)

        X = torch.from_numpy(signal[np.newaxis,]) # Only one channel
        y = torch.tensor(int(s))
        return {"x": X, "y": y}


class StratifiedSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements in a stratified manner, exhausting all posible files, "without" replacement.
    Arguments:
        dataset (Dataset): dataset to sample from
    """

    def __init__(self, groups: np.ndarray = required, shuffle=True):
        check_required(self, {'groups' : groups})
        # Store inputs
        self.N = groups.size
        self.shuffle = shuffle
        self.group_tags = np.unique(groups)
        self.groups = {g : np.where(groups == g)[0] for g in self.group_tags}
        
        # Auxiliary variables
        self.G = len(self.groups)
        self.counter_G = np.zeros((len(self.groups),),dtype=int)
        
    def __iter__(self):
        if self.shuffle:
            self.__shuffle()
        
        self.i = 0
        self.g = 0
        return self
    
    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        
        g = self.group_tags[self.i%self.G]
        r = (self.i//self.G)%self.groups[g].size
        index = self.groups[g][r]
        self.i += 1
        
        return index
    
    def __shuffle(self):
        for g in self.groups:
            self.groups[g] = np.random.permutation(self.groups[g])

    def __len__(self):
        return self.N



class StratifiedSamplerWFDB(torch.utils.data.sampler.Sampler):
    r"""Samples elements in a stratified manner, exhausting all posible files, "without" replacement.
    Arguments:
        dataset (Dataset): dataset to sample from
    """

    def __init__(self, dataset: torch.utils.data.Dataset, limit_N: int = None, maximum_N_per_file: int = 1e7, shuffle: bool = True):
        # Load dataset information
        self.dataset = dataset
        
        # Load max beat limitation
        self.limit_N = limit_N
        
        # Load maximum total number of beats
        self.maximum_N_per_file = maximum_N_per_file
        
        # Load shuffling options
        self.shuffle = shuffle
        
        # Create iterator
        self.__iterator = NoMemoryIterator(self.dataset.beat_distribution, limit_N=self.limit_N, shuffle=self.shuffle)
        
    def __iter__(self):
        return NoMemoryIterator(self.dataset.beat_distribution, limit_N=self.limit_N, maximum=self.maximum_N_per_file, shuffle=self.shuffle)

    def __len__(self):
        return len(self.__iterator)


class NoMemoryIterator:
    """Class to implement an iterator that does not rely on storing data in memory"""

    def __init__(self, n_beats: dict, limit_N: int = None, maximum: int = 1e7, shuffle: bool = True):
        # Shuffle option
        self.shuffle = shuffle

        # Store inputs
        self.n_beats = n_beats
        self.limit_N = limit_N
        
        # Declare maxima for stopping iterator
        self.all_symbols = list(self.n_beats.keys())
        self.max_symbols = len(self.all_symbols)
        self.all_files = [list(self.n_beats[self.all_symbols[i]].keys()) for i in range(self.max_symbols)]
        self.max_files = [len(self.all_files[i]) for i in range(self.max_symbols)]
        self.max_beats = [[n_beats[self.all_symbols[i]][self.all_files[i][j]] for j in range(self.max_files[i])] for i in range(self.max_symbols)]

        if self.limit_N is not None:
            if isinstance(self.limit_N, int):
                self.total_beats = max(max([[min([self.limit_N,self.n_beats[self.all_symbols[i]][self.all_files[i][j]]])*len(self.max_beats[i])*len(self.max_beats) for j in range(len(self.max_beats[i]))] for i in range(len(self.max_beats))]))
            elif isinstance(self.limit_N, str):
                if self.limit_N.lower() != 'auto':
                    raise ValueError("limit_N set to {} but currently only 'auto' allowed")

                number_of_beats = []
                factor_per_beat = []

                for i in range(len(self.max_beats)):
                    for j in range(len(self.max_beats[i])):
                        number_of_beats.append(self.n_beats[self.all_symbols[i]][self.all_files[i][j]])
                        factor_per_beat.append(len(self.max_beats[i])*len(self.max_beats))

                number_of_beats = np.array(number_of_beats)
                factor_per_beat = np.array(factor_per_beat)

                # limit total number of beats per epoch
                for i in range(0,number_of_beats.max(),100):
                    if (number_of_beats.clip(0,i)*factor_per_beat).max() > maximum:
                        break

                self.total_beats = (number_of_beats.clip(0,i)*factor_per_beat).max()
                self.limit_N = i
            else:
                raise TypeError("Invalid type association of limit_N")
        else:
            self.total_beats = max(max([[self.n_beats[self.all_symbols[i]][self.all_files[i][j]]*len(self.max_beats[i])*len(self.max_beats) for j in range(len(self.max_beats[i]))] for i in range(len(self.max_beats))]))
            

        # Declare counters
        self.counter_total = 0
        self.file = np.array([0 for _ in self.n_beats])
        self.symbol = 0
        
        # See if vectors can be avoided for arbitrarily large iteration
        self.ix_symbol = np.arange(self.max_symbols)
        self.ix_file = [np.arange(self.max_files[i]) for i in range(self.max_symbols)]
        
        if shuffle:
            self.ix_symbol = np.random.permutation(self.ix_symbol)
            self.ix_file = [np.random.permutation(self.ix_file[i]) for i in range(self.max_symbols)]
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.total_beats
    
    def __next__(self):
        if self.counter_total >= self.total_beats:
            raise StopIteration
        
        # Element added
        self.counter_total += 1

        # Sample symbol
        s = self.ix_symbol[self.symbol]
        # Sample file
        f = self.ix_file[s][self.file[s]]
        # Sample beat
        b = np.random.randint(0,self.max_beats[s][f])
        
        # First, we will be iterating over symbols to spread samples as much as possible
        self.symbol += 1
        if self.symbol == self.max_symbols:
            if self.shuffle:
                self.ix_symbol = np.random.permutation(self.ix_symbol)
            
            # Restart symbol
            self.symbol = 0

            # If all symbols have been iterated, go over next file for all
            self.file += 1
            
            # If reached max, start cycle again
            for i in range(self.max_symbols):
                if self.file[i] == self.max_files[i]:
                    self.file[i] = 0
                    
                    if self.shuffle:
                        self.ix_file[i] = np.random.permutation(self.ix_file[i])
        
        return (s,f,b)


