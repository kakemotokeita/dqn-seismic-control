import urllib.request
import numpy as np
import openseespy.opensees as op
import torch

FREE = 0
FIXED = 1

X = 1
Y = 2
ROTZ = 3

class Analysis:

    def __init__(self):
        # AIが取れるアクションの設定
        self.action = np.array([0, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        self.naction = len(self.action)

        self.beta = 1/4

        # 1質点系モデル
        self.T0 = 4
        self.h = self.action[0]
        self.hs = [self.h]
        self.m = 100
        self.k = 4*np.pi**2*self.m/self.T0**2

        # 入力地震動
        self.dt = 0.02
        to_meter = 0.01  # cmをmに変換する値
        self.wave_url = 'https://github.com/kakemotokeita/dqn-seismic-control/blob/master/wave/sample.csv'
        with urllib.request.urlopen(self.wave_url) as wave_file:
            self.wave_data = np.loadtxt(wave_file, usecols=(0,), delimiter=',', skiprows=3)*to_meter

        # OpenSees設定
        op.wipe()
        op.model('basic', '-ndm', 2, '-ndf', 3)  # 2 dimensions, 3 dof per node

        # 節点
        self.bot_node = 1
        self.top_node = 2
        op.node(self.bot_node, 0., 0.)
        op.node(self.top_node, 0., 0.)

        # 境界条件
        op.fix(self.top_node, FREE, FIXED, FIXED)
        op.fix(self.bot_node, FIXED, FIXED, FIXED)
        op.equalDOF(1, 2, *[Y, ROTZ])

        # 質量
        op.mass(self.top_node, self.m, 0., 0.)

        # 弾性剛性
        elastic_mat_tag = 1
        Fy = 1e10
        E0 = self.k
        b = 1.0
        op.uniaxialMaterial('Steel01', elastic_mat_tag, Fy, E0, b)

        # Assign zero length element
        beam_tag = 1
        op.element('zeroLength', beam_tag, self.bot_node, self.top_node, "-mat", elastic_mat_tag, "-dir", 1, '-doRayleigh', 1)

        # Define the dynamic analysis
        load_tag_dynamic = 1
        pattern_tag_dynamic = 1

        self.values = list(-1 * self.wave_data)  # should be negative
        op.timeSeries('Path', load_tag_dynamic, '-dt', self.dt, '-values', *self.values)
        op.pattern('UniformExcitation', pattern_tag_dynamic, X, '-accel', load_tag_dynamic)

        # 減衰の設定
        self.w0 = op.eigen('-fullGenLapack', 1)[0] ** 0.5
        self.alpha_m = 0.0
        self.beta_k = 2 * self.h / self.w0
        self.beta_k_init = 0.0
        self.beta_k_comm = 0.0

        op.rayleigh(self.alpha_m, self.beta_k, self.beta_k_init, self.beta_k_comm)

        # Run the dynamic analysis

        op.wipeAnalysis()

        op.algorithm('Newton')
        op.system('SparseGeneral')
        op.numberer('RCM')
        op.constraints('Transformation')
        op.integrator('Newmark', 0.5, 0.25)
        op.analysis('Transient')

        tol = 1.0e-10
        iterations = 10
        op.test('EnergyIncr', tol, iterations, 0, 2)
        self.i_pre = 0
        self.i = 0
        self.i_next = 0
        self.time = 0
        self.analysis_time = (len(self.values) - 1) * self.dt
        self.dis = 0
        self.vel = 0
        self.acc = 0
        self.a_acc = 0
        self.force = 0
        self.resp = {
            "time": [],
            "dis": [],
            "acc": [],
            "a_acc": [],
            "vel": [],
            "force": [],
        }
        self.done = False

    def reset(self):
        self.__init__()

    def step(self, action=0):
        self.time = op.getTime()
        assert(self.time < self.analysis_time)

        # 選ばれたアクションに応じて減衰定数を変化させる
        self.h = self.action[action]
        self.hs.append(self.h)
        self.beta_k = 2 * self.h / self.w0
        op.rayleigh(self.alpha_m, self.beta_k, self.beta_k_init, self.beta_k_comm)

        op.analyze(1, self.dt)
        op.reactions()

        self.dis = op.nodeDisp(self.top_node, 1)
        self.vel = op.nodeVel(self.top_node, 1)
        self.acc = op.nodeAccel(self.top_node, 1)
        self.a_acc = self.acc + self.values[self.i]
        self.force = -op.nodeReaction(self.bot_node, 1) # Negative since diff node

        self.resp["time"].append(self.time)
        self.resp["dis"].append(self.dis)
        self.resp["vel"].append(self.vel)
        self.resp["acc"].append(self.acc)
        self.resp["a_acc"].append(self.a_acc)
        self.resp["force"].append(self.force)

        next_time = op.getTime()
        self.done = next_time >= self.analysis_time

        self.i_pre = self.i
        self.i += 1
        self.i_next = self.i + 1 if not self.done else self.i
        return self.reward, self.done

    # 報酬
    @property
    def reward(self):
        return (10 / np.abs(self.a_acc))**3

    # 選ばれた減衰の平均値(参考値)
    @property
    def h_ave(self):
        return np.average(self.hs)

    # 選ばれた減衰の分散(参考値)
    @property
    def h_sd(self):
        return np.sqrt(np.var(self.hs))

    # 振動解析の現在の状態
    @property
    def state(self):
        return np.array([self.values[self.i_pre], self.values[self.i], self.values[self.i_next], self.a_acc, self.acc, self.vel, self.dis], dtype=np.float32)

    # 平均値
    @property
    def sd(self):
        return np.sqrt(np.var(np.abs(self.resp["a_acc"]))), np.sqrt(np.var(np.abs(self.resp["dis"])))

    # 最大値
    @property
    def max(self):
        return np.max(np.abs(self.resp["a_acc"])), np.max(np.abs(self.resp["dis"]))
