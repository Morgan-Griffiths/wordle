import datetime
import pathlib

import torch


class Config:
    seed = 1234
    gamma = 0.99
    gradient_clip = 10
    tau = 0.01
    eps = 0.5
    SGD_epoch = 4
    L2 = 0.1
    update_every = 4
    learning_update = 0
    # PER BUFFER
    batch_size = 64
    num_unroll_steps = 6
    replay_buffer_size = 1000
    PER_alpha = 0.1
    PER = True
    # MUZERO
    save_model = True
    checkpoint_interval = 10
    discount_rate = 0.85
    epsilon = 0.5
    action_space = 5
    value_loss_weight = 0.25
    weight_decay = 0.1
    lr_init = 1e-3
    lr_decay_rate = 0.999
    pb_c_base = 19652
    pb_c_init = 1.25
    root_exploration_fraction = 0.25
    root_dirichlet_alpha = 0.25
    num_simulations = 10
    revisit_policy_search_rate = 0
    self_play_delay = 0  # Number of seconds to wait after each played game
    training_delay = 0  # Number of seconds to wait after each training step
    temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
    num_workers = 1
    training_steps = 250
    td_steps = 6
    reanalyse_on_gpu = True
    selfplay_on_gpu = True
    train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
    use_last_model_value = True
    results_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "wordle"
        / "weights"
        / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    )  # Path to store the model weights and TensorBoard logs

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25
