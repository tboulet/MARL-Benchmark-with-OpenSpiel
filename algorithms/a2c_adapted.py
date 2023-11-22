from open_spiel.python.pytorch.policy_gradient import PolicyGradient


class PolicyGradient_Adapted(PolicyGradient):
    
    def __init__(self, 
        player_id,
        num_actions,
        state_representation_size,
        **kwargs,
        ):

        super().__init__(
            player_id = player_id, 
            num_actions = num_actions, 
            info_state_size = state_representation_size,  # this is the (only...) change needed
            **kwargs,
            )