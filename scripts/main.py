import math
import random
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from Analysis import Analysis
from DQN import DQN
from ReplayMemory import ReplayMemory
from Transition import Transition

# 振動解析を環境としてインスタンス化
env = Analysis()

# GPUが使えるかどうか
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"use {device}")


def get_state():
    return Variable(torch.from_numpy(env.state)).unsqueeze(0).unsqueeze(0)

# DQNをインスタンス化するためのサイズ取得
init_state = get_state()
_, _, state_size = init_state.size()

n_actions = env.naction     # 選択できるアクションの数

policy_net = DQN(state_size, n_actions, device).to(device)  # 方策を求めるためのネットワーク
target_net = DQN(state_size, n_actions, device).to(device)  # 最適化対象のネットワーク
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # 推論モードにする

# 最適化アルゴリズムにはRMSpropを選択
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


BATCH_SIZE = 128    # 複数の結果をまとめてニューラルネットワークに入力、分析する際のバッチサイズ
GAMMA = 0.5       # 遠い側の未来を考慮する割合（0に近いほど近い未来に重きをおく）

# ランダムのアクションを選択する閾値計算用の係数
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 800

TARGET_UPDATE = 100  # target_netを更新するエピソードの間隔

steps_done = 0

# あるstateでアクションを選択する関数
def select_action(state, test):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if test:
        with torch.no_grad():
            return target_net(state).max(1)[1].view(1, 1)
    elif sample > eps_threshold:
        with torch.no_grad():
            # 最も効果的と思われるアクションのインデックス
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # ランダムなアクションのインデックス
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# モデルの最適化
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # memoryからBATCH_SIZE分だけランダムにサンプルを取得
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # 解析終了時のステップ以外かどうかのBooleanとその時のnext_state
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 実際に取ったアクションの価値（実際に取って得られた報酬）
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # まだ更新されていないTarget_netによる最も大きい報酬
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # 本来期待されたアクションの価値
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss（実際に取ったアクションの価値と、本来期待されたアクションの価値を比較して損失を計算）
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1).type(torch.FloatTensor).to(device))

    # モデルの最適化
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 500
for i_episode in range(num_episodes + 1):
    # 環境の初期化
    env.reset()
    state = get_state()

    test = i_episode % TARGET_UPDATE == 0

    if test:
        # target_netを更新。全ての重みやバイアスをコピー
        target_net.load_state_dict(policy_net.state_dict())

    for t in count():
        # アクションを選択
        action = select_action(state, test)
        reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        next_state = get_state() if not done else None

        # memoryにTrasitionを記録
        memory.push(state, action, next_state, reward)

        state = next_state

        # モデル最適
        optimize_model()

        if done:
            acc_sd, dis_sd = env.sd
            acc_max, dis_max = env.max
            print('{0:3}'.format(str(i_episode)), 'h_ave=', '{0:4.3f}'.format(env.h_ave), 'h_sd=', '{0:4.3f}'.format(env.h_sd), 'acc_sd=', '{0:4.3f}'.format(acc_sd), 'dis_sd=', '{0:4.3f}'.format(dis_sd), 'acc_max=', '{0:4.3f}'.format(acc_max), 'dis_max=', '{0:4.3f}'.format(dis_max), 'test=', '{0:5}'.format(str(test)))
            break

print('Complete')
