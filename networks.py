"""Networks and rollout buffer for the three credit-assignment methods.

The COMA critic exists in two explicitly selectable versions:

``output_heads``  the reference implementation.  One network with one output per
                  joint action (64 outputs); only the taken joint action is
                  trained.  Counterfactual Q-values are untrained extrapolations
                  from the shared trunk.
``action_input``  one network Q(s, a) with the joint action one-hot concatenated
                  to the state.  The parameters are shared across all joint
                  actions, so a gradient on the taken action trains the function
                  that is also used to evaluate counterfactual actions.

Select the version with ``run_experiments.py --coma-critic {output_heads,action_input}``.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """One decentralized policy network for one drone."""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, x):
        logits = self.forward(x)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            distribution.probs,
        )

    def evaluate(self, x, actions):
        distribution = Categorical(logits=self.forward(x))
        return distribution.log_prob(actions), distribution.entropy()


class CentralizedCritic(nn.Module):
    """V(joint observation)."""

    def __init__(self, joint_obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs).squeeze(-1)


class JointActionCriticBase(nn.Module):
    """Shared interface of the two COMA critic parameterisations.

    Subclasses implement ``q_taken``, ``counterfactual_advantage`` and
    ``expected_q``; ``coma_target`` is shared.
    """

    def __init__(self, joint_obs_dim, n_agents, n_actions, hidden=64):
        super().__init__()
        self.joint_obs_dim = joint_obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_joint_actions = n_actions ** n_agents

    def joint_index(self, actions):
        """Map (batch, n_agents) action indices to a flat joint-action index."""
        index = torch.zeros(
            actions.shape[0], dtype=torch.long, device=actions.device
        )
        for i in range(self.n_agents):
            index = index * self.n_actions + actions[:, i]
        return index

    def _joint_probs(self, probs):
        """Product of the independent per-agent policies, in joint-index order."""
        batch = probs.shape[0]
        joint = probs[:, 0, :]
        for i in range(1, self.n_agents):
            joint = torch.einsum("ba,bc->bac", joint, probs[:, i, :]).reshape(
                batch, -1
            )
        return joint

    def q_taken(self, joint_obs, actions):
        raise NotImplementedError

    def counterfactual_advantage(self, joint_obs, actions, probs):
        raise NotImplementedError

    def expected_q(self, joint_obs, probs):
        raise NotImplementedError

    def coma_target(self, rewards, dones, next_joint_obs, next_probs, gamma):
        """r + gamma * E_{a'}[Q(s', a')] * (1 - done)."""
        expected = self.expected_q(next_joint_obs, next_probs)
        return rewards + gamma * expected * (1.0 - dones)

    def td_lambda_targets(
        self, joint_obs, actions, rewards, dones, gamma, lam
    ):
        """TD(lambda) return of the taken joint action (sampled next action).

        G_t = Q(s_t, a_t) + sum_k (gamma*lam)^k delta_{t+k},
        delta_t = r_t + gamma Q(s_{t+1}, a_{t+1}) (1 - done_t) - Q(s_t, a_t).
        The trace resets at episode boundaries.
        """
        length = rewards.shape[0]
        q_taken = self.q_taken(joint_obs, actions)
        q_next = torch.zeros(length, device=joint_obs.device)
        if length > 1:
            q_next[:-1] = self.q_taken(joint_obs[1:], actions[1:])
        deltas = rewards + gamma * q_next * (1.0 - dones) - q_taken
        targets = torch.zeros(length, device=joint_obs.device)
        trace = 0.0
        for t in reversed(range(length)):
            terminal = dones[t].item() == 1.0
            if terminal:
                trace = 0.0
            trace = deltas[t].item() + gamma * lam * (
                0.0 if terminal else 1.0
            ) * trace
            targets[t] = q_taken[t].item() + trace
        return targets


class JointActionCritic(JointActionCriticBase):
    """COMA critic, ``output_heads`` version (reference implementation).

    ``Q(s, .)`` for every joint action in one forward pass.  Only the taken
    joint action receives a gradient, so the counterfactual values are
    extrapolations from the shared trunk.
    """

    def __init__(self, joint_obs_dim, n_agents, n_actions, hidden=64):
        super().__init__(joint_obs_dim, n_agents, n_actions, hidden)
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.n_joint_actions),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs)

    def q_taken(self, joint_obs, actions):
        q = self.forward(joint_obs)
        base = self.joint_index(actions)
        return q.gather(1, base.unsqueeze(1)).squeeze(1)

    def counterfactual_advantage(self, joint_obs, actions, probs):
        """A_i = Q(s, a) - sum_{a_i'} pi_i(a_i'|o_i) Q(s, (a_i', a_-i))."""
        q = self.forward(joint_obs)
        base = self.joint_index(actions)
        q_taken = q.gather(1, base.unsqueeze(1)).squeeze(1)
        advantages = []
        for i in range(self.n_agents):
            stride = self.n_actions ** (self.n_agents - 1 - i)
            offsets = torch.arange(self.n_actions, device=q.device) * stride
            taken = actions[:, i] * stride
            indices = (
                base.unsqueeze(1) - taken.unsqueeze(1) + offsets.unsqueeze(0)
            )
            q_alt = q.gather(1, indices)
            advantages.append(
                q_taken - (probs[:, i, :] * q_alt).sum(dim=-1)
            )
        return torch.stack(advantages, dim=-1)

    def expected_q(self, joint_obs, probs):
        q = self.forward(joint_obs)
        return (self._joint_probs(probs) * q).sum(dim=-1)


class ActionInputCritic(JointActionCriticBase):
    """COMA critic, ``action_input`` version.

    ``Q(s, a)`` takes the joint observation and the one-hot joint action.  The
    network parameters are shared across all joint actions, so training on the
    taken action also trains the function used to evaluate counterfactual
    actions.
    """

    def __init__(self, joint_obs_dim, n_agents, n_actions, hidden=64):
        super().__init__(joint_obs_dim, n_agents, n_actions, hidden)
        self.action_dim = n_agents * n_actions
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim + self.action_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        one_hot = torch.zeros(self.n_joint_actions, self.action_dim)
        for j in range(self.n_joint_actions):
            for i in range(self.n_agents):
                stride = self.n_actions ** (self.n_agents - 1 - i)
                a = (j // stride) % self.n_actions
                one_hot[j, i * self.n_actions + a] = 1.0
        self.register_buffer("action_one_hot", one_hot, persistent=False)

    def one_hot(self, joint_indices):
        return self.action_one_hot[joint_indices]

    def forward(self, joint_obs, action_one_hot):
        x = torch.cat([joint_obs, action_one_hot], dim=-1)
        return self.net(x).squeeze(-1)

    def q_taken(self, joint_obs, actions):
        base = self.joint_index(actions)
        return self.forward(joint_obs, self.one_hot(base))

    def counterfactual_advantage(self, joint_obs, actions, probs):
        batch = joint_obs.shape[0]
        base = self.joint_index(actions)
        q_taken = self.forward(joint_obs, self.one_hot(base))
        advantages = []
        for i in range(self.n_agents):
            stride = self.n_actions ** (self.n_agents - 1 - i)
            offsets = torch.arange(
                self.n_actions, device=joint_obs.device
            ) * stride
            taken = actions[:, i] * stride
            indices = (
                base.unsqueeze(1) - taken.unsqueeze(1) + offsets.unsqueeze(0)
            )
            obs_exp = (
                joint_obs.unsqueeze(1)
                .expand(-1, self.n_actions, -1)
                .reshape(-1, self.joint_obs_dim)
            )
            act_exp = self.one_hot(indices.reshape(-1))
            q_alt = self.forward(obs_exp, act_exp).view(batch, self.n_actions)
            advantages.append(
                q_taken - (probs[:, i, :] * q_alt).sum(dim=-1)
            )
        return torch.stack(advantages, dim=-1)

    def expected_q(self, joint_obs, probs, chunk=256):
        """E_{a' ~ prod pi}[Q(s', a')] with Q evaluated for all joint actions."""
        batch = joint_obs.shape[0]
        joint_probs = self._joint_probs(probs)
        q_all = torch.zeros(
            batch, self.n_joint_actions, device=joint_obs.device
        )
        for start in range(0, batch, chunk):
            obs_chunk = joint_obs[start : start + chunk]
            size = obs_chunk.shape[0]
            obs_exp = (
                obs_chunk.unsqueeze(1)
                .expand(-1, self.n_joint_actions, -1)
                .reshape(-1, self.joint_obs_dim)
            )
            act_exp = (
                self.action_one_hot.unsqueeze(0)
                .expand(size, -1, -1)
                .reshape(-1, self.action_dim)
            )
            q_all[start : start + chunk] = self.forward(
                obs_exp, act_exp
            ).view(size, self.n_joint_actions)
        return (joint_probs * q_all).sum(dim=-1)


class RolloutBuffer:
    """Stores on-policy transitions for all drones."""

    def __init__(
        self, T, n_agents, obs_dim, n_actions, gamma=0.95, lam=0.95
    ):
        self.T = T
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        T, n, d, n_actions = (
            self.T,
            self.n_agents,
            self.obs_dim,
            self.n_actions,
        )
        self.obs = torch.zeros(T, n, d)
        self.actions = torch.zeros(T, n, dtype=torch.long)
        self.logprobs = torch.zeros(T, n)
        self.all_probs = torch.zeros(T, n, n_actions)
        self.team_rewards = torch.zeros(T)
        self.diff_rewards = torch.zeros(T, n)
        self.per_alternative = torch.zeros(T, n, n_actions)
        self.dones = torch.zeros(T)
        self.adv = torch.zeros(T, n)
        self.ret = torch.zeros(T, n)
        self.ptr = 0

    @property
    def joint_obs(self):
        return self.obs[: self.ptr].reshape(self.ptr, -1)

    def add(
        self,
        per_agent_obs,
        actions,
        logprobs,
        all_probs,
        team_reward,
        per_alt,
        done,
    ):
        if self.ptr >= self.T:
            raise RuntimeError("Rollout buffer is full")

        self.obs[self.ptr] = torch.as_tensor(
            per_agent_obs, dtype=torch.float32
        )
        self.actions[self.ptr] = torch.as_tensor(
            actions, dtype=torch.long
        )
        self.logprobs[self.ptr] = torch.as_tensor(
            logprobs, dtype=torch.float32
        )
        self.all_probs[self.ptr] = torch.as_tensor(
            all_probs, dtype=torch.float32
        )
        self.team_rewards[self.ptr] = float(team_reward)
        if per_alt is not None:
            self.per_alternative[self.ptr] = torch.as_tensor(
                per_alt, dtype=torch.float32
            )
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def compute_difference_rewards(self):
        """D_i = r_team - E_{a_i~pi_i}[r_team(a_i, a_-i)].

        The expectation uses the policy probabilities saved during rollout;
        the other drones' sampled actions remain fixed in each counterfactual.
        """
        if self.ptr == 0:
            return self.diff_rewards[:0]

        expected_counterfactual = (
            self.all_probs[: self.ptr] * self.per_alternative[: self.ptr]
        ).sum(dim=-1)
        self.diff_rewards[: self.ptr] = (
            self.team_rewards[: self.ptr].unsqueeze(-1)
            - expected_counterfactual
        )
        return self.diff_rewards[: self.ptr]

    def _gae_one(self, rewards, values, last_value, dones):
        length = rewards.shape[0]
        advantages = torch.zeros(length)
        last_advantage = 0.0

        for t in reversed(range(length)):
            next_value = (
                float(last_value) if t == length - 1 else values[t + 1].item()
            )
            nonterminal = 1.0 - dones[t].item()
            delta = (
                rewards[t].item()
                + self.gamma * next_value * nonterminal
                - values[t].item()
            )
            last_advantage = (
                delta
                + self.gamma
                * self.lam
                * nonterminal
                * last_advantage
            )
            advantages[t] = last_advantage

        return advantages, advantages + values

    def compute_gae_shared(self, value_predictions, last_value=0.0):
        advantages, returns = self._gae_one(
            self.team_rewards[: self.ptr],
            value_predictions[: self.ptr],
            last_value,
            self.dones[: self.ptr],
        )
        self.adv[: self.ptr] = advantages.unsqueeze(-1).expand(
            -1, self.n_agents
        )
        self.ret[: self.ptr] = returns.unsqueeze(-1).expand(
            -1, self.n_agents
        )
        return self.adv[: self.ptr], self.ret[: self.ptr]

    def compute_gae_difference(
        self, value_predictions, last_values_per_agent
    ):
        if self.ptr:
            if value_predictions.shape != (self.ptr, self.n_agents):
                raise ValueError(
                    "value_predictions must have shape (ptr, n_agents)"
                )
            if len(last_values_per_agent) != self.n_agents:
                raise ValueError("Need one bootstrap value per drone")

        for i in range(self.n_agents):
            advantages, returns = self._gae_one(
                self.diff_rewards[: self.ptr, i],
                value_predictions[: self.ptr, i],
                last_values_per_agent[i],
                self.dones[: self.ptr],
            )
            self.adv[: self.ptr, i] = advantages
            self.ret[: self.ptr, i] = returns

        return self.adv[: self.ptr], self.ret[: self.ptr]
