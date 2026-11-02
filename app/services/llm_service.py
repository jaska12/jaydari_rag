import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover - optional at runtime when adapter disabled
    PeftConfig = None
    PeftModel = None

from app.core.config import Settings


class LocalLLM:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        model, tokenizer = self._load_model_and_tokenizer(settings)
        device = 0 if settings.llm_device.lower() != 'cpu' and torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    def generate(self, prompt: str) -> str:
        generator = self._pipeline
        outputs = generator(
            prompt,
            max_new_tokens=self._settings.llm_max_new_tokens, # length
            temperature=self._settings.llm_temperature,
            do_sample=self._settings.llm_temperature > 0, # sampling enabled | creative responses
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        text = outputs[0]['generated_text'] # prompt + generated_text
        return text[len(prompt) :].strip()  # generated_text

    def _load_model_and_tokenizer(self, settings: Settings):
        adapter_path = settings.llm_adapter_path
        if not adapter_path:
            tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
            model = AutoModelForCausalLM.from_pretrained(settings.llm_model)
            return model, tokenizer

        if PeftModel is None or PeftConfig is None:
            raise RuntimeError(
                'PEFT adapter requested but "peft" is not installed. '
                'Install dependencies from requirements.txt.'
            )

        base_model_name = settings.llm_model
        try:
            peft_config = PeftConfig.from_pretrained(adapter_path)
            if peft_config.base_model_name_or_path:
                base_model_name = peft_config.base_model_name_or_path
        except Exception:
            peft_config = None

        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(model, adapter_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        return model, tokenizer
