import gymnasium as gym

# sys.path.append(".")
from src.v2.sac.sac import (
    NormalPolicyFunction,
    NormalPolicyFunctionConfig,
    SoftActorCritic,
    SoftActorCriticConfig,
    ValueFunction,
    ValueFunctionConfig,
)

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    # use same config as for TF1 implementation
    config = SoftActorCriticConfig(
        q_fct_config=ValueFunctionConfig(),
        v_fct_config=ValueFunctionConfig(),
        pi_fct_config=NormalPolicyFunctionConfig(hidden_layers=[100, 100], output_activation_function_log_std=None),
        discount=0.99,
        tau=0.01,
        batch_size=256,
        alpha=0.1,
        learning_rate_v=3e-4,
        learning_rate_pi=3e-4,
        learning_rate_q=3e-4,
        learning_rate_alpha=3e-4,
        buffer_size=int(1e4),
        max_grad_norm=None,
    )

    # set up the trainer
    sac_agent = SoftActorCritic(
        config,
        policy_function=NormalPolicyFunction,
        value_function=ValueFunction,
        env=env,
    )

    # train networks
    r = sac_agent.train(
        epochs=50, max_steps=500, env_steps=1, grad_steps=1, n_burn_in_steps=1000, n_log_epochs=1, log_output="stdout"
    )

    # evaluate how well the policy network performs visually
    sac_agent.run_agent_on_env(gym.make("Pendulum-v1", render_mode="human"), max_steps=500)
