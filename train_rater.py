import argparse
import json
import os
import re
import sys

from datasets import load_dataset
from unsloth import (
    FastLanguageModel,
    FastModel,
    UnslothTrainer,
    is_bfloat16_supported,
)
from unsloth.chat_templates import (
    get_chat_template,
    train_on_responses_only,
)

# isort: off
from transformers import TextStreamer
from trl import SFTConfig

# Constants
MAX_SEQ_LENGTH = 2**16 - 2**13
LOAD_IN_4BIT = True
LORA_RANK = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MICRO_BATCH_SIZE = 2
ACCUM_STSEPS = 4
MODEL = "gemma"

if MODEL == "gemma":
    CHAT_TEMPLATE = "gemma-3"
    MODEL_ARCH = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
    instruction_part = "<start_of_turn>user\n"
    response_part = "<start_of_turn>model"
    response_regex = r"<start_of_turn>model\s(.*)<end_of_turn>"
elif MODEL == "phi-4":
    CHAT_TEMPLATE = "Phi-4"
    MODEL_ARCH = "unsloth/Phi-4"
    instruction_part = "<|im_start|>user<|im_sep|>"
    response_part = "<|im_start|>assistant<|im_sep|>"
    response_regex = r"<\|im_start\|>assistant<\|im_sep\|>(.*)<\|im_end\|>"
else:
    print("Invalid model", MODEL)
    sys.exit(1)


SYSTEM_PROMPT = (
    "You are a world-class securities analyst and hedge fund manager with a proven track record of outperforming the market using long and short strategies. "
    "Given information on a publicly traded company, your task is to conduct a comprehensive analysis of company's prospects and predict the stock's future "
    "performance. You will assign a rating on a scale from 1 to 5 indicating your recommendation to buy (4-5), ignore (3), or sell (1-2) the stock. "
    "You are detail-oriented and contrarian. Given that the market as a whole goes up over time, you have a positive bias. But you are not afraid to take a bearish stance "
    "when the data supports it - there are many scams, pumps, liars, fake news, and overvalued companies. "
)

PROMPT_APPEND = '\nProvide your answer as a JSON object of the form: {"entity": str, "rating": float, "returns_90d": float, "returns_180d": float, "returns_365d": float}'


def load_and_prepare_dataset(split, tokenizer, is_test=False):
    dataset = load_dataset(
        path="data", data_files=[f"{split}.jsonl.zst"], split="train"
    )

    def format_answer(t):
        return f'{{"entity": "{t["entity"]}", "rating": {t["rating"]}, "returns_30d": {t["returns_30d"]}, "returns_90d": {t["returns_90d"]}, "returns_180d": {t["returns_180d"]}, "returns_365d": {t["returns_365d"]}}}'

    def formatting_prompts_func(examples):
        if is_test:
            messages = []
            for input in examples["input"]:
                messages.append(
                    [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": input + PROMPT_APPEND}
                            ],
                        },
                    ]
                )

            return {"text": messages}

        convos = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": input + PROMPT_APPEND}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": format_answer(json.loads(target))}
                    ],
                },
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
                t = t[-MAX_SEQ_LENGTH:]
            out.append(t)

        return {"text": out, "output": examples["output"]}

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=["input"] if is_test else ["input", "output"],
    )
    return dataset


def train_model(out_dir):
    global tokenizer  # Needed for formatting function

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        full_finetuning=False,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)

    dataset = load_and_prepare_dataset("train", tokenizer)
    #  val_dataset = load_and_prepare_dataset("val", tokenizer)

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # SHould leave on always!
        r=LORA_RANK,  # Larger = higher accuracy, but might overfit
        lora_alpha=LORA_RANK,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        # eval_dataset=val_dataset
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            dataset_text_field="text",
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=ACCUM_STSEPS,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=WEIGHT_DECAY,
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
        instruction_part=instruction_part,
        response_part=response_part,
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
    model, tokenizer = FastModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    # test_dataset = load_and_prepare_dataset("test", tokenizer, is_test=True)
    val_dataset = load_and_prepare_dataset("val", tokenizer, is_test=True)

    out_rows = []

    for item in val_dataset:
        convo = item["text"]

        input_ids = tokenizer.apply_chat_template(
            convo,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        # text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        output = model.generate(
            input_ids=input_ids,
            # streamer=text_streamer,
            max_new_tokens=256,
            use_cache=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )

        output = tokenizer.batch_decode(output)[0]
        out = re.search(response_regex, output)
        if out is not None:
            output = out.group(1)
        else:
            print("Bad output:", output)
            pass

        try:
            out_rows.append(
                json.loads(output.replace("```json", "").replace("```", ""))
            )
            print("---")
            print(out_rows[-1])
            print("===")
            print(item["output"])

        except json.JSONDecodeError:
            print("Bad output:", output)

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
            break
