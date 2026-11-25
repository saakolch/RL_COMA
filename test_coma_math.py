"""Unit tests for both COMA critic versions against brute-force computations.

Run:
    python test_coma_math.py

The two critic versions are:
    JointActionCritic  (output_heads)  one output per joint action
    ActionInputCritic  (action_input)  Q(s, a) with the joint action as input
Both must produce the same counterfactual advantage and target for the same
Q-values, and those values must match the formulas evaluated by hand.
"""
import torch

from networks import ActionInputCritic, JointActionCritic

N_AGENTS = 3
N_ACTIONS = 4
N_JOINT = N_ACTIONS ** N_AGENTS


def all_q(critic, joint_obs):
    """Q(s, .) for all joint actions, for either critic version."""
    if isinstance(critic, JointActionCritic):
        return critic(joint_obs)
    batch = joint_obs.shape[0]
    obs_exp = joint_obs.unsqueeze(1).expand(-1, N_JOINT, -1).reshape(
        -1, critic.joint_obs_dim
    )
    act_exp = (
        critic.action_one_hot.unsqueeze(0)
        .expand(batch, -1, -1)
        .reshape(-1, critic.action_dim)
    )
    return critic.forward(obs_exp, act_exp).view(batch, N_JOINT)


def brute_advantage(q, actions, probs):
    batch = q.shape[0]
    out = torch.zeros(batch, N_AGENTS)
    for b in range(batch):
        a = actions[b].tolist()
        base = a[0] * 16 + a[1] * 4 + a[2]
        q_taken = q[b, base].item()
        for i in range(N_AGENTS):
            alternatives = []
            for alt in range(N_ACTIONS):
                aa = list(a)
                aa[i] = alt
                alternatives.append(aa[0] * 16 + aa[1] * 4 + aa[2])
            expected = sum(
                probs[b, i, alt].item() * q[b, alternatives[alt]].item()
                for alt in range(N_ACTIONS)
            )
            out[b, i] = q_taken - expected
    return out


def check(critic_name, critic):
    torch.manual_seed(0)
    joint_obs = torch.randn(6, critic.joint_obs_dim)
    actions = torch.tensor(
        [[0, 0, 0], [1, 2, 3], [3, 3, 3], [2, 1, 0], [0, 3, 1], [1, 1, 2]]
    )
    probs = torch.softmax(torch.randn(6, N_AGENTS, N_ACTIONS), dim=-1)

    assert torch.equal(
        critic.joint_index(actions)[:3], torch.tensor([0, 27, 63])
    )

    q = all_q(critic, joint_obs)
    advantage = critic.counterfactual_advantage(joint_obs, actions, probs)
    expected = brute_advantage(q, actions, probs)
    assert torch.allclose(advantage, expected, atol=1e-5), (
        advantage, expected
    )

    expected_q = critic.expected_q(joint_obs, probs)
    expected_q_brute = (critic._joint_probs(probs) * q).sum(dim=-1)
    assert torch.allclose(expected_q, expected_q_brute, atol=1e-5)

    rewards = torch.tensor([1.0, 2.0, 0.5, -1.0, 0.0, 3.0])
    dones = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    next_obs = torch.randn(6, critic.joint_obs_dim)
    next_probs = torch.softmax(torch.randn(6, N_AGENTS, N_ACTIONS), dim=-1)
    target = critic.coma_target(rewards, dones, next_obs, next_probs, 0.9)
    next_q = all_q(critic, next_obs)
    expected_target = rewards + 0.9 * (
        (critic._joint_probs(next_probs) * next_q).sum(dim=-1)
    ) * (1.0 - dones)
    assert torch.allclose(target, expected_target, atol=1e-5)

    q_taken = critic.q_taken(joint_obs, actions)
    base = critic.joint_index(actions)
    assert torch.allclose(q_taken, q.gather(1, base.unsqueeze(1)).squeeze(1))
    print(f"{critic_name}: advantage, expected_q, target, q_taken OK")


def one_hot_exactness():
    """A deterministic policy must reproduce the exact counterfactual value."""
    joint_obs = torch.zeros(1, 10)
    actions = torch.tensor([[0, 0, 0]])
    probs = torch.zeros(1, N_AGENTS, N_ACTIONS)
    probs[0, 0, 2] = 1.0
    probs[0, 1, 1] = 1.0
    probs[0, 2, 3] = 1.0
    for critic in (
        JointActionCritic(10, N_AGENTS, N_ACTIONS),
        ActionInputCritic(10, N_AGENTS, N_ACTIONS),
    ):
        q = all_q(critic, joint_obs)
        advantage = critic.counterfactual_advantage(joint_obs, actions, probs)
        # with one-hot policies the expectation is Q of the alternative action
        alt0 = q[0, 2 * 16].item()
        alt1 = q[0, 1 * 4].item()
        alt2 = q[0, 3].item()
        assert abs(advantage[0, 0].item() - (q[0, 0].item() - alt0)) < 1e-5
        assert abs(advantage[0, 1].item() - (q[0, 0].item() - alt1)) < 1e-5
        assert abs(advantage[0, 2].item() - (q[0, 0].item() - alt2)) < 1e-5
    print("one-hot counterfactual exactness OK")


def td_lambda_check():
    """TD(lambda) targets must match the forward-view return."""
    critic = JointActionCritic(10, N_AGENTS, N_ACTIONS)
    torch.manual_seed(1)
    length = 12
    joint_obs = torch.randn(length, critic.joint_obs_dim)
    actions = torch.randint(0, N_ACTIONS, (length, N_AGENTS))
    dones = torch.zeros(length)
    dones[5] = 1.0
    dones[length - 1] = 1.0
    rewards = torch.randn(length)
    gamma, lam = 0.9, 0.8

    q = critic.q_taken(joint_obs, actions)
    q_next = torch.zeros(length)
    q_next[:-1] = critic.q_taken(joint_obs[1:], actions[1:])
    deltas = rewards + gamma * q_next * (1.0 - dones) - q

    brute = torch.zeros(length)
    for t in range(length):
        acc = 0.0
        k = 0
        for u in range(t, length):
            acc += (gamma * lam) ** k * deltas[u].item()
            k += 1
            if dones[u].item() == 1.0:
                break
        brute[t] = q[t].item() + acc

    target = critic.td_lambda_targets(
        joint_obs, actions, rewards, dones, gamma, lam
    )
    assert torch.allclose(target, brute, atol=1e-5), (target, brute)
    print("td_lambda_targets OK")


def main():
    check("output_heads", JointActionCritic(10, N_AGENTS, N_ACTIONS))
    check("action_input", ActionInputCritic(10, N_AGENTS, N_ACTIONS))
    one_hot_exactness()
    td_lambda_check()
    print("all COMA math tests passed")


if __name__ == "__main__":
    main()
