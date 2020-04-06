import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt
import utils
import hyp
from dqn import MolDQN
from rdkit import Chem
from rdkit.Chem import QED
from environment import Molecule
from baselines.deepq import replay_buffer

REPLAY_BUFFER_CAPACITY = hyp.replay_buffer_size


class QEDRewardMolecule(Molecule):
    """The molecule whose reward is the QED."""

    def __init__(self, discount_factor, **kwargs):
        """Initializes the class.

    Args:
      discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
      **kwargs: The keyword arguments passed to the base class.
    """
        super(QEDRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor

    def _reward(self):
        """Reward of a state.

    Returns:
      Float. QED of the current state.
    """
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        qed = QED.qed(molecule)
        return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)


class Agent(object):
    def __init__(self, input_length, output_length, device):
        self.device = device
        self.dqn, self.target_dqn = (
            MolDQN(input_length, output_length).to(self.device),
            MolDQN(input_length, output_length).to(self.device),
        )
        for p in self.target_dqn.parameters():
            p.requires_grad = False
        self.replay_buffer = replay_buffer.ReplayBuffer(REPLAY_BUFFER_CAPACITY)
        self.optimizer = getattr(opt, hyp.optimizer)(
            self.dqn.parameters(), lr=hyp.learning_rate
        )

    def get_action(self, observations, epsilon_threshold):

        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn.forward(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).numpy()

        return action

    def update_params(self, batch_size, gamma, polyak):
        # update target network

        # sample batch of transitions
        states, _, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        q_t = torch.zeros(batch_size, 1, requires_grad=False)
        v_tp1 = torch.zeros(batch_size, 1, requires_grad=False)
        for i in range(batch_size):
            state = (
                torch.FloatTensor(states[i])
                .reshape(-1, hyp.fingerprint_length + 1)
                .to(self.device)
            )
            q_t[i] = self.dqn(state)

            next_state = (
                torch.FloatTensor(next_states[i])
                .reshape(-1, hyp.fingerprint_length + 1)
                .to(self.device)
            )
            v_tp1[i] = torch.max(self.target_dqn(next_state))

        rewards = torch.FloatTensor(rewards).reshape(q_t.shape).to(self.device)
        q_t = q_t.to(self.device)
        v_tp1 = v_tp1.to(self.device)
        dones = torch.FloatTensor(dones).reshape(q_t.shape).to(self.device)

        # # get q values
        q_tp1_masked = (1 - dones) * v_tp1
        q_t_target = rewards + gamma * q_tp1_masked
        td_error = q_t - q_t_target

        q_loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error * td_error,
            1.0 * (torch.abs(td_error) - 0.5),
        )
        q_loss = q_loss.mean()

        # backpropagate
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return q_loss
