from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, List

import torch
from torch.nn import CrossEntropyLoss
from torch import nn

from transformers import DistilBertForMaskedLM, DistilBertConfig, DistilBertPreTrainedModel, PretrainedConfig, \
    add_start_docstrings
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutput, TokenClassifierOutput
from transformers.models.distilbert.modeling_distilbert import Transformer

from transformers.utils import ModelOutput, add_start_docstrings_to_model_forward, add_code_sample_docstrings

PROTEOMELM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DistilBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PROTEOMELM_INPUTS_DOCSTRING = r"""
    Args:
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        group_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Embeddings for the group of sequences.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

_CHECKPOINT_FOR_DOC = "proteomelm"
_CONFIG_FOR_DOC = "ProteomeLMConfig"


def polarize(x):
    x_norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    x_polar = x / x_norm
    return x_polar


class ProteomeLMConfig(DistilBertConfig):
    """
    This is the configuration class to store the configuration of a [`ProteomeLMModel`]. It is used to
    instantiate a ProteomeLM model according to the specified arguments, defining the model architecture.
    Args:"""

    def __init__(
            self,
            input_size: int = 640,
            loss_type: str = "mse",
            **kwargs,
    ):
        self.input_size = input_size
        self.loss_type = loss_type
        super().__init__(**kwargs)


@dataclass
class ProteomeLMMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        norm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Loss for the norm of the predicted direction.
        cosine_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Loss for the cosine similarity of the predicted direction.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        prediction_scores (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        prediction_norm (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Prediction scores of the norm head (scores for each vocabulary token before SoftMax).
        ppi_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Prediction scores of the PPI head (scores for each PPI before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        last_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    norm_loss: Optional[torch.FloatTensor] = None
    cosine_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    prediction_scores: Optional[torch.FloatTensor] = None
    prediction_norm: Optional[torch.FloatTensor] = None
    ppi_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@add_start_docstrings(
    "The bare ProteomeLM encoder/transformer outputting raw hidden-states without any specific head on top.",
    PROTEOMELM_INPUTS_DOCSTRING,
)
class ProteomeLMModel(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.transformer = Transformer(config)  # Encoder
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.embeddings.position_embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[List[int]]]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PROTEOMELM_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        head_mask_is_none = head_mask is None
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embeddings = inputs_embeds  # (bs, seq_length, dim)

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

            if self._use_sdpa and head_mask_is_none and not output_attentions:
                attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embeddings.dtype, tgt_len=input_shape[1]
                )

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class ProteomeLMForMaskedLM(DistilBertForMaskedLM):
    _keep_keys_on_attention_maps_predictions = ["inputs_embeds", "adversaries", "group_embeds"]

    def __init__(self, config, ):
        super().__init__(config)
        self.embedding_main = nn.Linear(config.input_size, config.dim)
        self.embedding_encoder = nn.Linear(config.input_size, config.dim)
        self.transformer = ProteomeLMModel(config)
        self.lm_head = nn.Linear(config.dim, config.input_size, bias=False)
        self.lm_norm = nn.Linear(config.dim, 1, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.loss_choice = "polar"  # mse, cosine, or mse+cosine
        assert self.loss_choice in ["mse", "cosine", "polar"], f"Loss type {self.loss_type} not supported."

        # Initialize weights and apply final processing
        self.post_init()

    def get_contextualized_embeddings(self,
                                      prediction_scores: torch.Tensor,
                                      prediction_norm: torch.Tensor,
                                      root: torch.Tensor):
        residue = prediction_scores - root
        return residue / torch.linalg.vector_norm(residue, ord=2, dim=-1, keepdim=True) * prediction_norm + root

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            group_embeds: Optional[torch.Tensor] = None,
            masked_tokens: Optional[torch.BoolTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

            *args, **kwargs,
    ) -> "ProteomeLMMaskedLMOutput":

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mask = None
        if group_embeds is None:
            group_embeds = inputs_embeds.clone()
        if masked_tokens is not None:
            mask = masked_tokens.bool()
            inputs_embeds[mask] = group_embeds[mask].clone()
            root = group_embeds[mask].clone()
        else:
            root = group_embeds.clone()
        inputs_embeds = self.embedding_main(inputs_embeds)
        group_embeds = self.embedding_encoder(group_embeds)
        inputs_embeds = inputs_embeds + group_embeds

        transformer_outputs = self.transformer(
            input_ids=None,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0].view(inputs_embeds.size(0), -1, self.config.dim)
        prediction_scores = self.lm_head(hidden_states)
        prediction_norm = self.lm_norm(hidden_states)
        if masked_tokens is not None:
            prediction_scores = prediction_scores[mask]
            prediction_norm = prediction_norm[mask]
            embeds = self.get_contextualized_embeddings(prediction_scores, prediction_norm, root)
            if not return_dict:
                return (prediction_scores, prediction_norm) + transformer_outputs[1:]
            return ProteomeLMMaskedLMOutput(
                loss=None,
                norm_loss=None,
                cosine_loss=None,
                logits=embeds,
                prediction_scores=prediction_scores,
                prediction_norm=prediction_norm,
                hidden_states=transformer_outputs.hidden_states,
                last_hidden_states=hidden_states,
                attentions=transformer_outputs.attentions,
            )
        embeds = self.get_contextualized_embeddings(prediction_scores, prediction_norm, root)
        all_hidden_states = None
        if output_hidden_states:
            all_hidden_states = []
            for h in transformer_outputs.hidden_states:
                h = h.view(h.size(0), -1, self.config.dim)
                all_hidden_states.append(h)

        if not return_dict:
            return (embeds, prediction_scores, prediction_norm) + transformer_outputs[1:]
        return ProteomeLMMaskedLMOutput(
            loss=None,
            logits=embeds,
            prediction_scores=prediction_scores,
            prediction_norm=prediction_norm,
            hidden_states=all_hidden_states,
            last_hidden_states=hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    ProteomeLM Model with a token classification head on top (a linear layer on top of the hidden-states output
    for each token) e.g. for Named-Entity Recognition (NER) tasks.
    """,
    PROTEOMELM_START_DOCSTRING,
)
class ProteomeLMForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.proteomelm = ProteomeLMModel(config)  # Base Transformer model (no token embeddings inside)
        self.embedding_main = nn.Linear(config.input_size, config.dim)
        self.embedding_encoder = nn.Linear(config.input_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.dim, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(PROTEOMELM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        group_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Prepare attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # If group_embeds is not provided, use inputs_embeds (or input_ids cast to float) as group_embeds
        if inputs_embeds is None:
            # If only input_ids were provided, we convert them to floats for embedding layers
            inputs_embeds = input_ids.to(device=device, dtype=torch.float32)
        if group_embeds is None:
            group_embeds = inputs_embeds.clone()

        # Compute input embeddings by projecting the input features to hidden dim and adding group embeddings
        inputs_embeds_main = self.embedding_main(inputs_embeds)
        inputs_embeds_enc = self.embedding_encoder(group_embeds)
        combined_embeds = inputs_embeds_main + inputs_embeds_enc

        # Forward pass through the ProteomeLM transformer
        outputs = self.proteomelm(
            input_ids=None,
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # The last hidden state is the first element of outputs (either a tuple or BaseModelOutput)
        sequence_output = outputs[0]  # shape: (batch_size, sequence_length, hidden_size)

        # Apply dropout and compute logits
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # shape: (batch_size, sequence_length, num_labels)

        loss = None
        if labels is not None:
            # Flatten the predictions and true labels for computing loss
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                # Only compute loss on active (non-padded) tokens
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # Return a tuple consistent with BaseModelOutput conventions
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # Return a TokenClassifierOutput (which is a ModelOutput, containing predictions and optional loss/hidden states/attentions)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
