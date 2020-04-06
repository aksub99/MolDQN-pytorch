import torch
from agent import Agent
from agent import QEDRewardMolecule, Agent
import hyp
import math
import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter

TENSORBOARD_LOG = True
TB_LOG_PATH = "./runs/dqn/run2"
episodes = 0
iterations = 200000
update_interval = 20
batch_size = 128
num_updates_per_it = 1

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

environment = QEDRewardMolecule(
    discount_factor=hyp.discount_factor,
    atom_types=set(hyp.atom_types),
    init_mol=hyp.start_molecule,
    allow_removal=hyp.allow_removal,
    allow_no_modification=hyp.allow_no_modification,
    allow_bonds_between_rings=hyp.allow_bonds_between_rings,
    allowed_ring_sizes=set(hyp.allowed_ring_sizes),
    max_steps=hyp.max_steps_per_episode,
)

# DQN Inputs and Outputs:
# input: appended action (fingerprint_length + 1) .
# Output size is (1).

agent = Agent(hyp.fingerprint_length + 1, 1, device)

if TENSORBOARD_LOG:
    writer = SummaryWriter(TB_LOG_PATH)

environment.initialize()

eps_threshold = 1.0
batch_losses = []

for it in range(iterations):

    steps_left = hyp.max_steps_per_episode - environment.num_steps_taken

    # Compute a list of all possible valid actions. (Here valid_actions stores the states after taking the possible actions)
    valid_actions = list(environment.get_valid_actions())

    # Append each valid action to steps_left and store in observations.
    observations = np.vstack(
        [
            np.append(
                utils.get_fingerprint(
                    act, hyp.fingerprint_length, hyp.fingerprint_radius
                ),
                steps_left,
            )
            for act in valid_actions
        ]
    )  # (num_actions, fingerprint_length)

    observations_tensor = torch.Tensor(observations)
    # Get action through epsilon-greedy policy with the following scheduler.
    # eps_threshold = hyp.epsilon_end + (hyp.epsilon_start - hyp.epsilon_end) * \
    #     math.exp(-1. * it / hyp.epsilon_decay)

    a = agent.get_action(observations_tensor, eps_threshold)

    # Find out the new state (we store the new state in "action" here. Bit confusing but taken from original implementation)
    action = valid_actions[a]
    # Take a step based on the action
    result = environment.step(action)

    action_fingerprint = np.append(
        utils.get_fingerprint(action, hyp.fingerprint_length, hyp.fingerprint_radius),
        steps_left,
    )

    next_state, reward, done = result

    # Compute number of steps left
    steps_left = hyp.max_steps_per_episode - environment.num_steps_taken

    # Append steps_left to the new state and store in next_state
    next_state = utils.get_fingerprint(
        next_state, hyp.fingerprint_length, hyp.fingerprint_radius
    )  # (fingerprint_length)

    action_fingerprints = np.vstack(
        [
            np.append(
                utils.get_fingerprint(
                    act, hyp.fingerprint_length, hyp.fingerprint_radius
                ),
                steps_left,
            )
            for act in environment.get_valid_actions()
        ]
    )  # (num_actions, fingerprint_length + 1)

    # Update replay buffer (state: (fingerprint_length + 1), action: _, reward: (), next_state: (num_actions, fingerprint_length + 1),
    # done: ()

    agent.replay_buffer.add(
        obs_t=action_fingerprint,  # (fingerprint_length + 1)
        action=0,  # No use
        reward=reward,
        obs_tp1=action_fingerprints,  # (num_actions, fingerprint_length + 1)
        done=float(result.terminated),
    )

    if done:
        final_reward = reward
        if episodes != 0 and TENSORBOARD_LOG and len(batch_losses) != 0:
            writer.add_scalar("episode_reward", final_reward, episodes)
            writer.add_scalar("episode_loss", np.array(batch_losses).mean(), episodes)
        if episodes != 0 and episodes % 2 == 0 and len(batch_losses) != 0:
            print(
                "reward of final molecule at episode {} is {}".format(
                    episodes, final_reward
                )
            )
            print(
                "mean loss in episode {} is {}".format(
                    episodes, np.array(batch_losses).mean()
                )
            )
        episodes += 1
        eps_threshold *= 0.99907
        batch_losses = []
        environment.initialize()

    if it % update_interval == 0 and agent.replay_buffer.__len__() >= batch_size:
        for update in range(num_updates_per_it):
            loss = agent.update_params(batch_size, hyp.gamma, hyp.polyak)
            loss = loss.item()
            batch_losses.append(loss)
