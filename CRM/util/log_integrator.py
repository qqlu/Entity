"""
Integrate numerical values for some iterations
Typically used for loss computation
Just call finalize and create a new Integrator when you want to display 
"""
class Integrator:
    def __init__(self, logger):
        self.values = {}
        self.counts = {}
        self.hooks  = [] # List is used here to maintain insertion order

        self.logger = logger

    def add_tensor(self, key, tensor):
        if key not in self.values:
            self.counts[key] = 1
            if type(tensor) == float or type(tensor) == int:
                self.values[key] = tensor
            else:
                self.values[key] = tensor.mean().item()
        else:
            self.counts[key] += 1
            if type(tensor) == float or type(tensor) == int:
                self.values[key] += tensor
            else:
                self.values[key] += tensor.mean().item()

    def add_dict(self, tensor_dict):
        for k, v in tensor_dict.items():
            self.add_tensor(k, v)

    def add_hook(self, hook):
        """
        Adds a custom hook, i.e. compute new metrics using values in the dict
        The hook takes the dict as argument, and returns a (k, v) tuple
        """
        if type(hook) == list:
            self.hooks.extend(hook)
        else:
            self.hooks.append(hook)

    def reset_except_hooks(self):
        self.values = {}
        self.counts = {}

    # Average and output the metrics
    def finalize(self, prefix, iter, f=None):

        for hook in self.hooks:
            k, v = hook(self.values)
            self.add_tensor(k, v)

        for k, v in self.values.items():
            avg = v / self.counts[k]

            self.logger.log_metrics(prefix, k, avg, iter, f)

