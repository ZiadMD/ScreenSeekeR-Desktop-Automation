"""
Inference utilities for GUI-Actor model.
Adapted from microsoft/GUI-Actor (MIT License).
"""

import torch
from transformers import LogitsProcessor


class ForceFollowTokensLogitsProcessor(LogitsProcessor):
    """
    Forces tokens B (pointer_pad_token) and C (pointer_end_token) to follow
    token A (pointer_start_token).
    
    Whenever token_a_id is generated, enqueue the forced_sequence (e.g. [B, C]).
    As long as forced tokens remain in the queue, force them in the output.
    """

    def __init__(self, token_a_id: int, forced_sequence: list):
        super().__init__()
        self.token_a_id = token_a_id
        self.forced_sequence = forced_sequence
        self.force_queue = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        if batch_size > 1:
            raise NotImplementedError("Batch size must be 1 for this logits processor.")

        last_token_id = input_ids[0, -1].item()

        # If the last token was A (pointer_start), enqueue the forced sequence
        if last_token_id == self.token_a_id:
            self.force_queue.extend(self.forced_sequence)

        # If we have forced tokens waiting, override the distribution
        if len(self.force_queue) > 0:
            forced_token = self.force_queue.pop(0)
            new_scores = torch.full_like(scores, float('-inf'))
            new_scores[0, forced_token] = 0.0  # log prob = 0 => prob = 1
            return new_scores

        return scores
