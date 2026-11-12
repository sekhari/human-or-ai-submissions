"""Tiny randomly initialized models: no pretrained downloads or GPU required."""
import sys
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2Model, T5Config, T5ForConditionalGeneration, RobertaConfig, RobertaForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformer_models import QwenClassifier, binary_logits
from scratch import original

torch.set_num_threads(2)
torch.manual_seed(42)
original.check_model(torch.device('cpu'))
base = Qwen2Model(Qwen2Config(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
adapter = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj']))
qwen = QwenClassifier(adapter)
t5 = T5ForConditionalGeneration(T5Config(vocab_size=40, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, dropout_rate=0.0))
t5.config.binary_label_ids = [5, 6]
roberta = RobertaForSequenceClassification(RobertaConfig(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_labels=2, pad_token_id=0))
batch = dict(input_ids=torch.tensor([[2, 3, 4, 0], [2, 7, 8, 9]]), attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]), labels=torch.tensor([0, 1]))
for model, kind in [(qwen, 'qwen'), (t5, 't5'), (roberta, 'roberta')]:
    logits = binary_logits(model, batch, kind)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, batch['labels']).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
assert all(p.grad is None for name, p in adapter.named_parameters() if 'lora_' not in name)
assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in adapter.named_parameters() if 'lora_' in name)
qwen.eval()
short = {k: v[:1, :3] for k, v in batch.items() if k != 'labels'}
with torch.no_grad():
    torch.testing.assert_close(qwen(**short).logits, qwen(**{k: v[:1] for k, v in batch.items() if k != 'labels'}).logits, atol=1e-6, rtol=1e-5)
print('BiGRU, RoBERTa, T5 and Qwen forward/backward, padding, and LoRA checks passed.')
