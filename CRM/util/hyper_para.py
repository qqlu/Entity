from argparse import ArgumentParser

class HyperParameters():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        # Generic learning parameters
        parser.add_argument('-i', '--iterations', help='Number of training iterations', default=4.5e4, type=int)
        parser.add_argument('-b', '--batch_size', help='Batch size', default=12, type=int)
        parser.add_argument('--lr', help='Initial learning rate', default=2.25e-4, type=float)
        parser.add_argument('--steps', help='Iteration at which learning rate is decayed by gamma', default=[22500, 37500], type=int, nargs='*')
        parser.add_argument('--gamma', help='Gamma used in learning rate decay', default=0.1, type=float)
        parser.add_argument('--weight_decay', help='Weight decay', default=1e-4, type=float)

        # same decay applied to discriminator
        parser.add_argument('--load', help='Path to pretrained model if available')

        parser.add_argument('--ce_weight', help='Weight of the CE loss', default=0.0, type=float)
        parser.add_argument('--l1_weight', help='Weight of the L1 loss', default=1.0, type=float)
        parser.add_argument('--l2_weight', help='Weight of the L2 loss', default=1.0, type=float)
        parser.add_argument('--grad_weight', help='Weight of the gradient loss', default=5.0, type=float)

        # Logging information, this one is positional and mandatory
        parser.add_argument('id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

    def __getitem__(self, key):
        return self.args[key]

    def __str__(self):
        return str(self.args)

