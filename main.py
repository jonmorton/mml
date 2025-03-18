import argparse
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    is_bfloat16_supported,
)
from unsloth.chat_templates import (
    standardize_sharegpt,
    get_chat_template,
    train_on_responses_only,
)
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer

# Constants
MAX_SEQ_LENGTH = 32768 * 2
LOAD_IN_4BIT = True
LORA_RANK = 32


def load_and_prepare_dataset(split, tokenizer, is_test=False):
    dataset = load_dataset("data", data_files=[split], split="train")
    dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        for convo in convos:
            for c in convo:
                c["content"] = c["content"][-MAX_SEQ_LENGTH:]
        if is_test:
            return convos
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def train_model():
    global tokenizer  # Needed for formatting function
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-4",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        max_lora_rank=LORA_RANK,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

    dataset = load_and_prepare_dataset("train.json", tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=1e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.1,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_steps=5,
            save_strategy="steps",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user<|im_sep|>",
        response_part="<|im_start|>assistant<|im_sep|>",
    )
    trainer.train()

    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    model.save_pretrained_merged("lora_model", tokenizer, save_method = "merged_16bit")
    model.save_pretrained_merged("lora_model", tokenizer, save_method = "lora")


def test_model(checkpoint):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)

    test_dataset = load_and_prepare_dataset("test.json", tokenizer, is_test=True)

    for item in test_dataset:
        input_ids = tokenizer.apply_chat_template(
            item, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            input_ids=input_ids,
            streamer=text_streamer,
            max_new_tokens=256,
            use_cache=True,
            temperature=1.0,
            min_p=0.1,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a language model.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Sub-parser for training
    train_parser = subparsers.add_parser("train", help="Train the language model.")

    # Sub-parser for testing
    test_parser = subparsers.add_parser("test", help="Test the language model.")
    test_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path for testing.")

    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        test_model(args.checkpoint)
