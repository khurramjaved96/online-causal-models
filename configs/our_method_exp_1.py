import configargparse


class Parser(configargparse.ArgParser):
    def __init__(self):
        super().__init__()
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--steps', type=int, nargs='+', default=[5010000])
        self.add('--no-noise', action='store_true')
        self.add('--rank', type=int, help='meta batch size, namely task num', default=0)
        self.add('--name', help='Name of experiment', default="online_v_estimate/")
        self.add('--output-dir', help='Name of experiment', default="../results/")
        self.add('--update-lr', nargs='+', type=float, help='task-level inner update learning rate',
                 default=[1e-4])
        self.add('--l1_penalty', type=float, nargs='+', help='meta-level outer learning rate', default=[1e-4])
        self.add('--alpha', type=float, nargs='+', help='meta-level outer learning rate', default=[1e-4])
        self.add('--less-likely', help='meta-level outer learning rate', default="20,21")
        self.add('--model-path', help='Type of model', default=None)
