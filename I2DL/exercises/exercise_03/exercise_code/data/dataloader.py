"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################
        # shuffle if needed
        if self.shuffle:
            randomized_index_list = np.random.permutation(len(self.dataset))
        else:
            randomized_index_list = range(len(self.dataset))
        # yield whole batches (those have exactly size batch_size)
        batch_num_floor = np.floor_divide(len(self.dataset), self.batch_size)
        for i in range(batch_num_floor):
            curr_idx_batch = randomized_index_list[i*self.batch_size: (i+1)*self.batch_size]
            yield {"data": np.array([self.dataset[idx]["data"] for idx in curr_idx_batch])}
        # yield last batch if needed
        if not self.drop_last and len(self.dataset)%self.batch_size != 0:
            curr_idx_batch = randomized_index_list[batch_num_floor*self.batch_size: len(self.dataset)-1]
            yield {"data": np.array([self.dataset[idx]["data"] for idx in curr_idx_batch])}
        pass
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last!                                 #
        ########################################################################
        length = np.floor_divide(len(self.dataset), self.batch_size)
        if not self.drop_last and len(self.dataset)%self.batch_size != 0: 
            length += 1
        pass
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
