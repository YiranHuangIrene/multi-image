import torch

class DelimiterTokenScaler:
    def __init__(self, model, scaling_factor=5.0, target_layers=None, delimiter_token_ids=None):
        """
        Args:
            model: The HuggingFace model (e.g., Qwen2.5-VL).
            scaling_factor (float): Lambda value from the paper.
            target_layers (list[int]): Indices of layers to apply scaling (e.g., [0, 1, 2, 3]).
            delimiter_token_ids (list[int]): IDs of tokens to scale.
        """
        self.model = model
        self.scaling_factor = scaling_factor
        self.target_layers = target_layers 
        self.delimiter_token_ids = set(delimiter_token_ids) if delimiter_token_ids else set()
        
        self.current_mask = None
        self.hooks = []

    def _top_level_input_hook(self, module, args, kwargs):
        """
        Intercepts the inputs to the model (before layers) to calculate the mask 
        dynamically for the current batch.
        """
        # HF models usually receive input_ids as the first arg or in kwargs
        input_ids = kwargs.get("input_ids", None) # shape (batch, seq_len)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
            
        if input_ids is not None:
            # Create a boolean mask: True where token is a delimiter
            # We use a loop for multiple delimiter IDs (usually just 2-3)
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in self.delimiter_token_ids:
                mask |= (input_ids == tid)
            
            self.current_mask = mask
        return args, kwargs

    def _layer_hook(self, module, args):
        """
        Scales hidden states in the specific transformer layers.
        """
        if self.current_mask is None:
            return args

        hidden_states = args[0]
        mask = self.current_mask.to(hidden_states.device)

        # Safety: Ensure shapes match (e.g., during decoding steps vs prefill)
        # During decoding, hidden_states might be (batch, 1, dim) while mask is (batch, seq_len)
        # We only care about scaling during the 'prefill' (prompt processing) where delimiters exist.
        if mask.shape[1] == hidden_states.shape[1]:
            # [cite_start]Apply scaling: h_d = lambda * h_d [cite: 211]
            # Clone to avoid in-place errors
            scaled_states = hidden_states.clone()
            scaled_states[mask] = hidden_states[mask] * self.scaling_factor
            return (scaled_states,) + args[1:]
        
        return args

    def register(self):
        self.remove_hooks()
        
        # 1. Hook the top-level model to capture input_ids
        self.hooks.append(self.model.register_forward_pre_hook(self._top_level_input_hook, with_kwargs=True))

        # 2. Hook the specific layers
        # Locate layers (works for Qwen2/LLaVA style architectures)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            layers = self.model.model.language_model.layers
        else:
            raise ValueError("Could not locate model layers.")

        for i in self.target_layers:
            if i < len(layers):
                self.hooks.append(layers[i].register_forward_pre_hook(self._layer_hook))
        
        print(f"Delimiter Scaling Active: Factor={self.scaling_factor}, Layers={self.target_layers}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    