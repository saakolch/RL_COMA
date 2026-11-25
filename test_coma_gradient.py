"""Compare how the two COMA critics receive gradient.

Run:
    python test_coma_gradient.py

Case A: every sample takes the same joint action.  Only that action's output row
is trained, which is why the counterfactual values at that state are not
directly supervised in a single update.

Case B: the batch covers all 64 joint actions (as a near-uniform policy does
over a rollout).  Every output row is then trained, but each row still sees only
the samples where its joint action was taken.

The ``action_input`` critic shares parameters across joint actions, so its
counterfactual evaluations come from the same trained function.
"""
import torch

from networks import ActionInputCritic, JointActionCritic

torch.manual_seed(0)


def joint_index(actions):
    index = torch.zeros(actions.shape[0], dtype=torch.long)
    for i in range(3):
        index = index * 4 + actions[:, i]
    return index


# ---------------------------------------------------------------- case A
critic = JointActionCritic(10, 3, 4)
joint_obs = torch.randn(8, 10)
actions = torch.zeros(8, 3, dtype=torch.long)
critic.q_taken(joint_obs, actions).mean().backward()
grad = critic.net[-1].weight.grad
nonzero = (grad.abs().sum(dim=1) > 1e-12).nonzero().flatten().tolist()
print("output_heads, case A (one joint action): rows with gradient:", nonzero)
print("output_heads, case A: gradient on other 63 rows:",
      grad[1:].abs().sum().item())

# ---------------------------------------------------------------- case B
critic = JointActionCritic(10, 3, 4)
all_actions = torch.tensor(
    [[(j // 16) % 4, (j // 4) % 4, j % 4] for j in range(64)]
)
joint_obs = torch.randn(64, 10)
critic.q_taken(joint_obs, all_actions).mean().backward()
grad = critic.net[-1].weight.grad
nonzero = (grad.abs().sum(dim=1) > 1e-12).nonzero().flatten().tolist()
print("output_heads, case B (all 64 joint actions): rows with gradient:",
      len(nonzero), "of", grad.shape[0])

# ---------------------------------------------------------------- action input
critic = ActionInputCritic(10, 3, 4)
actions = torch.zeros(8, 3, dtype=torch.long)
critic.q_taken(torch.randn(8, 10), actions).mean().backward()
last_grad = critic.net[-1].weight.grad
first_grad = critic.net[0].weight.grad
print("action_input: output-layer rows with gradient:",
      int((last_grad.abs().sum(dim=1) > 1e-12).sum().item()))
print("action_input: first-layer parameters with gradient:",
      int((first_grad.abs().sum(dim=1) > 1e-12).sum().item()), "of",
      first_grad.shape[0])
print("action_input: counterfactual evaluations use the same trained parameters")
