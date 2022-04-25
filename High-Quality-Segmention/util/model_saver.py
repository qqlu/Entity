import os
import torch

class ModelSaver:
    def __init__(self, id):

        if id is None:
            self.no_log = True
            print('Saving has been disbaled.')
        else:
            self.no_log = False

            self.save_path = os.path.join('.', 'weights', '%s' % id )

    def save_model(self, model, step):
        if self.no_log:
            print('Saving has been disabled.')
            return

        os.makedirs(self.save_path, exist_ok=True)

        model_path = os.path.join(self.save_path, 'model_%s' % step)
        torch.save(model.state_dict(), model_path)
        print('Model saved to %s.' % model_path)
