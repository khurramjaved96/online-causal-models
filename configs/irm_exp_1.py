import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('--gpus', type=int, help='meta-level outer learning rate', default=1)
        self.add('--no-gpu', action='store_true')
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--steps', type=int, nargs='+', default=[5010000])
        self.add('--no-noise', action='store_true')
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--name', help='Name of experiment', default="IRM_baseline/")
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add('--update-lr', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[1e-4])
        self.add('--l1_penalty', type=float, nargs='+', help='meta-level outer learning rate', default=[1e-4])
        self.add('--grad_penalty', type=float, nargs='+', help='meta-level outer learning rate', default=[100000])
        self.add('--less-likely', help='meta-level outer learning rate', default="20,21")
