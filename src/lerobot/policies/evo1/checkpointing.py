"""Instance-local adaptation of InternVL3's reentrant-only vision checkpoint loop."""

from types import MethodType

from torch.utils.checkpoint import checkpoint


def _forward(self, inputs_embeds, output_hidden_states=None, return_dict=None):
    from transformers.modeling_outputs import BaseModelOutput

    output_hidden_states = (
        self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
    )
    return_dict = self.config.use_return_dict if return_dict is None else return_dict
    states = () if output_hidden_states else None
    hidden = inputs_embeds
    for layer in self.layers:
        if output_hidden_states:
            states += (hidden,)
        if self.gradient_checkpointing and self.training:
            hidden = checkpoint(layer, hidden, use_reentrant=False)
        else:
            hidden = layer(hidden)
    if output_hidden_states:
        states += (hidden,)
    if not return_dict:
        return tuple(x for x in (hidden, states) if x is not None)
    return BaseModelOutput(last_hidden_state=hidden, hidden_states=states)


def install_non_reentrant_vision_checkpoint(encoder):
    """Preserve parameters and state keys; do not patch torch or cached remote source."""
    encoder.forward = MethodType(_forward, encoder)
