class DataSet:
    '''A Naive Type of Data. Only takes boolean and numbers as content.'''
    def __init__(self, path=None, y=None, dataset=None, indexes=None):
        if path == None:  # initialize with dataset and selected indexes to create subset
            self.y = dataset.y
            self.attributes = dataset.attributes
            self.dataset = {index : dataset.dataset[index] for index in indexes}
            self._i = 0
        else:
            with open(path, 'r') as f:
                lines = f.readlines()

            self.y = y
            self.attributes = lines[0].split()
            self.attributes.remove(y)
            self.dataset = {}
            self.parse_dataset(lines)
            self._i = 0

    def parse_dataset(self, lines):
        for index in range(1, len(lines)):
            self.dataset[index] = {}
            for j in range(len(lines[0].split())):
                attr = lines[0].split()[j]
                raw_val = lines[index].split()[j]
                if raw_val == 'TRUE' or raw_val == 'true' or raw_val == 'True':
                    self.dataset[index][attr] = True
                elif raw_val == 'FALSE' or raw_val == 'false' or raw_val == 'False':
                    self.dataset[index][attr] = False
                else:
                    self.dataset[index][attr] = float(raw_val)

    def __iter__(self):
        return self

    def next(self):
        if self._i >= len(self.dataset.keys()):
            self._i = 0
            raise StopIteration
        else:
            self._i += 1
            index = self.dataset.keys()[self._i - 1]
            return {index : self.dataset[index]}

    def indexes(self):
        return self.dataset.keys()

    def __getitem__(self, name):
        ## if name == index, return the dict in that index
        ## if name == indexes, return a new DataSet constructed of corresponding dicts
        try:  # name == indexes
            iter(name)
            return DataSet(dataset=self, indexes=name)
        except:  # name == index
            return self.dataset[name]

    def get_vals(self, attr):
        return [self.dataset[index][attr] for index in self.indexes()]

    def append(self, entry):
        # entry is one {index : attributes}
        key = entry.keys()[0]
        self.dataset[key] = entry[key]