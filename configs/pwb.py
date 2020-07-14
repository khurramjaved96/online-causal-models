import configargparse


class PwbParameters(configargparse.ArgParser):
    def __init__(self):
        super().__init__()

        self.add('--gpus', type=int, help='Total GPUs in the system', default=1)
        self.add('--no-gpu', action='store_true', help='Do not use CPUs', default=True)
        self.add('--seed', nargs='+', help='Seed', default=[90], type=int)
        self.add('--features', nargs='+', type=int, default=[100])
        self.add('--steps', type=int, help="Total training steps", nargs='+',default=[40000])
        self.add('--rank', type=int, help='Rank to specify hyper-parameter when if multiple values are passed for the same parameter', default=0)
        self.add('--env1', type=float, help='Value of latent variable in one part of the seen MDP', default=0.01)
        self.add('--env2', type=float, help='Value of the latent variable in the other part of the seen MDP', default=0.24)
        self.add('--name', help='Name of experiment', default="PwB/")
        self.add('--output-dir', help='Directory where results and trained model is stored', default="../results/")
        self.add('--l1-penalty', type=float, nargs='+', help='L1 reguarlization', default=[1e-3])
