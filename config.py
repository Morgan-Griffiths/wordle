class Config:
    seed = 1234
    gamma = 0.99
    gradient_clip = 10
    tau = 0.01
    eps = 0.5
    buffer_size = 1000
    batch_size = 32
    SGD_epoch = 4
    alpha = 0.1
    L2 = 0.1
    update_every = 4
    learning_update = 0
    PER = True
    # MUZERO
    value_loss_weight = 0.25
    weight_decay = 0.1
    lr_init = 1e-3
    pb_c_base = 19652
    pb_c_init = 1.25
    root_exploration_fraction = 0.25
    root_dirichlet_alpha = 0.25
    num_simulations = 10
    revisit_policy_search_rate = 0
