"""Qwen3-VL image-aware collator; labels supervise ONLY the assistant answer.

Run processor individually to preserve image_grid_thw and pixel_values packing.
Avoid splitting a multimodal sequence at max_length (would invalidate grid alignment).
"""
import torch


PROMPT = "Answer the question based on the document image. Give only the short answer. Question: {question}"


def messages_for(image, question, answer=None):
    result = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT.format(question=question)},
    ]}]
    if answer is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return result


class DocVQACollator:
    def __init__(self, processor, max_length=4096, train=True):
        self.processor = processor
        self.max_length = max_length
        self.train = train
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id")

    def _encode(self, messages, generation_prompt):
        out = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=generation_prompt,
            return_dict=True, return_tensors="pt",
        )
        out.pop("token_type_ids", None)
        return out

    def __call__(self, examples):
        encoded = []
        for ex in examples:
            image, question = ex["image"], ex["question"]
            prompt = self._encode(messages_for(image, question), True)
            if self.train:
                answer = ex["answer"]
                if not answer:
                    raise ValueError("Training requires reference answers")
                full = self._encode(messages_for(image, question, answer), False)
                # Prompt template is intended to be exact prefix of the full text.
                prefix = prompt["input_ids"].shape[-1]
                if not torch.equal(full["input_ids"][0, :prefix], prompt["input_ids"][0]):
                    raise ValueError("Assistant prefix mismatch; check transformers chat template version")
                ids = full["input_ids"].squeeze(0)
                labels = ids.clone()
                labels[:prefix] = -100
            else:
                full = prompt
                ids = full["input_ids"].squeeze(0)
                labels = None
            if ids.numel() > self.max_length:
                raise ValueError(
                    f"Example {ex['question_id']} has {ids.numel()} tokens > {self.max_length}. "
                    "Lower max_pixels or increase max_length; do not truncate images."
                )
            encoded.append((full, ids, labels))

        longest = max(ids.numel() for _, ids, _ in encoded)
        bsz = len(encoded)
        input_ids = torch.full((bsz, longest), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, longest), dtype=torch.long)
        labels_batch = torch.full((bsz, longest), -100, dtype=torch.long) if self.train else None
        other = {}
        for i, (out, ids, labels) in enumerate(encoded):
            n = len(ids)
            input_ids[i, :n] = ids
            attention_mask[i, :n] = 1
            if self.train:
                labels_batch[i, :n] = labels
            for key, value in out.items():
                if key in ("input_ids", "attention_mask"):
                    continue
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Unexpected processor field {key}: {type(value)}")
                other.setdefault(key, []).append(value)
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.train:
            result["labels"] = labels_batch
        for key, values in other.items():
            # Qwen3-VL image_grid_thw = [images,3], pixel_values = [patches,dim].
            result[key] = torch.cat(values, dim=0)
        return result
