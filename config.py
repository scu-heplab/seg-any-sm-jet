import random


class Config:
    def __init__(self):
        # 环境参数
        self.MTL = 0.189
        self.rlitz = 0.71e-3
        self.N = 160
        self.Rac = 1.68e-8 * self.MTL * self.N / (3.14 * self.rlitz ** 2) * 1.3
        self.Ae = 859e-6
        self.Bm = 400 * 50e-6 / (self.Ae * self.N)
        self.Pcore = 0
        self.Pwinding = 0
        self.hcond = 83.3 / 62e-3
        self.S = 5.7e-4
        self.Rhs = 1 / (self.hcond * self.S)
        self.p0 = 0.5
        self.deltaTwinding = 0
        self.U2rate = 400
        self.U1rate = 400

        self.Ls = 400e-6

        self.lamad = 0.02
        self.Updiv0 = 1150  # -20%~0
        self.deltaPDIV = 200  # ±20%
        self.deltaPDIVf = 20
        self.lamda_f = 1 / 10e3
        self.lamda_f2 = 1 / 10e3
        self.kdeltaf = 0.05
        self.alpha = 1.5  # ±30%
        self.delta = 0.005
        self.yita = 0.8  # ±20%
        self.kpdq = 1
        self.Urate = 1000
        self.otherthing = 0
        self.ivevportion = 0

        # 动作参数
        self.u_scale = 10
        self.p_scale = 100

        # 算法参数
        self.device = "cuda"
        self.max_step = 64
        self.memory = 10000
        self.epoch = 8
        self.trajectory = 8
        self.batch_size = 128

        self.state_dim = 16
        self.action_dim = 5
        self.hidden_dim = 64

        self.lr = 3e-4
        self.tau = 0.01
        self.gamma = 0.99
        self.clip_param = 0.2
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.max_episode = 10000

        self.run_name = "PPO2"

    def reset(self, seed=None):
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
        else:
            state = None
        self.Updiv0 = random.uniform(1150 - 230, 1150)  # -20%~0
        self.deltaPDIV = random.uniform(200 - 40, 200 + 40)  # ±20%
        self.alpha = random.uniform(1.5 - 0.45, 1.5 + 0.45)  # ±30%
        self.yita = random.uniform(0.8 - 0.16, 0.8 + 0.16)  # ±20%
        if state is not None:
            random.setstate(state)
