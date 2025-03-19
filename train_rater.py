import argparse
import json
import os
import re
import sys
from math import e

from datasets import load_dataset
from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    is_bfloat16_supported,
)
from unsloth.chat_templates import (
    get_chat_template,
    standardize_sharegpt,
    train_on_responses_only,
)

# isort: off
from transformers import DataCollatorForSeq2Seq, TextStreamer, TrainingArguments

# Constants
MAX_SEQ_LENGTH = 2**16
LOAD_IN_4BIT = True
LORA_RANK = 16

SYSTEM_PROMPT = (
    "You are a world-class securities analyst and hedge fund manager with a proven track record of outperforming the market using long and short strategies. "
    "Given information on a publicly traded company, your task is to conduct a comprehensive analysis of company's prospects and predict the stock's future "
    "performance. You will assign a rating on a scale from 1 to 5 indicating your recommendation to buy (4-5), ignore (3), or sell (1-2) the stock. "
    "You are detail-oriented and contrarian. Given that the market as a whole goes up over time, you have a positive bias. But you are not afraid to take a bearish stance "
    "when the data supports it - there are many scams, pumps, liars, fake news, and overvalued companies. "
)


def load_and_prepare_dataset(split, tokenizer, is_test=False):
    dataset = load_dataset(
        path="data", data_files=[f"{split}.jsonl.zst"], split="train"
    )

    def formatting_prompts_func(examples):
        if is_test:
            messages = []
            for input in examples["input"]:
                messages.append(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": input},
                    ]
                )

            return {
                "text": messages,
            }

        convos = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input},
                {"role": "assistant", "content": target},
            ]
            for input, target in zip(examples["input"], examples["output"])
        ]

        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]

        out = []
        for t in texts:
            if len(t) > MAX_SEQ_LENGTH:
                print("Truncated prompt from {} to {}".format(len(t), MAX_SEQ_LENGTH))
                print(t)
                t = t[-MAX_SEQ_LENGTH:]
            out.append(t)

        return {"text": out}

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=["input"] if is_test else ["input", "output"],
    )
    return dataset


def train_model(out_dir):
    global tokenizer  # Needed for formatting function
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-4",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        max_lora_rank=LORA_RANK,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

    dataset = load_and_prepare_dataset("train", tokenizer)
    #  val_dataset = load_and_prepare_dataset("val", tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0,  # Supports any, but = 0 is optimized
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        # eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.04,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=out_dir,
            save_steps=5,
            save_strategy="steps",
            report_to="wandb",
            # do_eval=True,
            # eval_steps=5,
            # eval_strategy="steps",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user<|im_sep|>",
        response_part="<|im_start|>assistant<|im_sep|>",
    )
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    model.save_pretrained_merged(
        os.path.join(out_dir, "merged_16bit"), tokenizer, save_method="merged_16bit"
    )
    model.save_pretrained_merged(
        os.path.join(out_dir, "lora"), tokenizer, save_method="lora"
    )


def test_model(checkpoint):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)

    test_dataset = load_and_prepare_dataset("test", tokenizer, is_test=True)
    out_rows = []

    for item in test_dataset:
        convo = item["text"]
        input_ids = tokenizer.apply_chat_template(
            convo, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        output = model.generate(
            input_ids=input_ids,
            streamer=text_streamer,
            max_new_tokens=256,
            use_cache=True,
            temperature=1.0,
            min_p=0.1,
            pad_token_id=tokenizer.pad_token_id,
        )

        output = tokenizer.batch_decode(output)[0]
        out = re.search(
            r"<\|im_start\|>\s*assistant\s*<\|im_sep\|>\s*(.*)\s*<\|im_end\|>", output
        )
        if out is not None:
            output = out.group(1)
        else:
            print("Bad output:", output)
            pass

        out_rows.append(json.loads(output))
        print(out_rows[-1])

    with open("output.json", "w") as f:
        json.dump(out_rows, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a language model.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Sub-parser for training
    train_parser = subparsers.add_parser("train", help="Train the language model.")
    train_parser.add_argument(
        "--out", type=str, required=True, help="Output directory for saving the model"
    )

    # Sub-parser for testing
    test_parser = subparsers.add_parser("test", help="Test the language model.")
    test_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint path for testing."
    )

    # Sub-parser for testing dataset
    test_dataset_parser = subparsers.add_parser(
        "test_dataset", help="Test the dataset formatting function."
    )
    test_dataset_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint path for testing."
    )

    args = parser.parse_args()

    if args.mode == "train":
        os.environ["WANDB_PROJECT"] = "train_rater"
        os.environ["WANDB_NAME"] = args.out

        train_model(args.out)
    elif args.mode == "test":
        test_model(args.checkpoint)
    elif args.mode == "test_dataset":
        # test dataset impl
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.checkpoint,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=LOAD_IN_4BIT,
        )
        dataset = load_and_prepare_dataset("test", tokenizer, is_test=True)

        for item in dataset:
            print(item)
            break
