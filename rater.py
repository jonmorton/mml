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
from trl import SFTConfig
from yaml import load

# Constants
MAX_SEQ_LENGTH = 2**16 - 2**13 - 512
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MICRO_BATCH_SIZE = 2
ACCUM_STEPS = 4
MODEL = "gemma"
LORA_RANK = 128

if MODEL == "gemma":
    CHAT_TEMPLATE = "gemma-3"
    instruction_part = "<start_of_turn>user\n"
    response_part = "<start_of_turn>model\n"
    response_regex = r"<start_of_turn>model\s(.*)<end_of_turn>"

    def MODEL_BUILDER(model_name="unsloth/gemma-3-1b-it"):
        return FastModel.from_pretrained(
            model_name=model_name, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=False, load_in_8bit=False
        )

    def MODEL_PEFT(model):
        return FastModel.get_peft_model(
            model,
            finetune_language_layers=True,  # Should leave on!
            finetune_attention_modules=True,  # Attention good for GRPO
            finetune_mlp_modules=True,  # SHould leave on always!
            finetune_vision_layers=False,
            r=LORA_RANK,  # Larger = higher accuracy, but might overfit
            lora_alpha=32,  # Recommended alpha == r at least
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            max_seq_length=MAX_SEQ_LENGTH,
        )

    def TO_INFERENCE(model):
        return FastModel.for_inference(model)

    def FORMAT_TXT(string):
        return [{"text": string, "type": "text"}]

elif MODEL == "phi-4":
    CHAT_TEMPLATE = "phi-4"
    instruction_part = "<|im_start|>user<|im_sep|>\n"
    response_part = "<|im_start|>assistant<|im_sep|>\n"
    response_regex = r"<\|im_start\|>\s*assistant\s*<\|im_sep\|>\s*(.*)<\|im_end\|>"

    def MODEL_BUILDER(model_name="unsloth/Phi-4"):
        return FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,
        )

    def MODEL_PEFT(model):
        return FastLanguageModel.get_peft_model(
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
            lora_alpha=32,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,
            max_seq_length=MAX_SEQ_LENGTH,
        )

    def TO_INFERENCE(model):
        return FastLanguageModel.for_inference(model)

    def FORMAT_TXT(string):
        return string
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

PROMPT_APPEND = '\nProvide your answer as a JSON object of the form: {"entity": str, "rating": float}'


def load_and_prepare_dataset(split, tokenizer, is_test=False):
    dataset = load_dataset(
        path="/workspace/data", data_files=[f"{split}.jsonl.zst"], split="train"
    )

    def format_answer(t):
        # return f'{{"entity": "{t["entity"]}", "rating": {t["rating"]}, "returns_30d": {t["returns_30d"]}, "returns_90d": {t["returns_90d"]}, "returns_180d": {t["returns_180d"]}, "returns_365d": {t["returns_365d"]}}}'
        return f'{{"entity": "{t["entity"]}", "rating": {round(t["rating"])}}}'

    def formatting_prompts_func(examples):
        if is_test:
            messages = []
            for input in examples["input"]:
                messages.append(
                    [
                        {"role": "system", "content": FORMAT_TXT(SYSTEM_PROMPT)},
                        {
                            "role": "user",
                            "content": FORMAT_TXT(input + PROMPT_APPEND),
                        },
                    ]
                )

            return {"text": messages}

        convos = [
            [
                {
                    "role": "system",
                    "content": FORMAT_TXT(SYSTEM_PROMPT),
                },
                {
                    "role": "user",
                    "content": FORMAT_TXT(input + PROMPT_APPEND),
                },
                {
                    "role": "assistant",
                    "content": FORMAT_TXT(format_answer(json.loads(target))),
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

    model, tokenizer = MODEL_BUILDER()
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    dataset = load_and_prepare_dataset("train", tokenizer)
    model = MODEL_PEFT(model)

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
            gradient_accumulation_steps=ACCUM_STEPS,
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
            save_steps=20,
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

    # model.save_pretrained_merged(
    #     os.path.join(out_dir, "merged_16bit"), tokenizer, save_method="merged_16bit"
    # )
    # model.save_pretrained_merged(
    #     os.path.join(out_dir, "lora"), tokenizer, save_method="lora"
    # )


def test_model(checkpoint):
    model, tokenizer = MODEL_BUILDER(checkpoint)
    model = TO_INFERENCE(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    # test_dataset = load_and_prepare_dataset("test", tokenizer, is_test=True)
    val_dataset = load_and_prepare_dataset("val", tokenizer, is_test=True)

    out_rows = []

    correct = incorrect = 0
    total_sqerr = 0
    tot_l1 = 0
    within1 = 0

    ratings = {}

    for item in val_dataset:
        convo = item["text"]

        input_ids = tokenizer.apply_chat_template(
            convo,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            use_cache=True,
            temperature=0.5,
            min_p=0.1,
            #top_p=0.95,
            #top_k=64,
        )

        output = tokenizer.batch_decode(output)[0]
        out = re.search(response_regex, output, re.DOTALL)
        if out is not None:
            output = out.group(1)
        else:
            print("Bad output:", output)
            pass

        try:
            out_rows.append(json.loads(output))
            out = round(out_rows[-1]["rating"])
            targ = round(json.loads(item["output"])["rating"])

            ratings[out] = ratings.get(out, 0) + 1

            print("Output:", out)
            print("Target:", targ)

            if abs(out - targ) < 0.5:
                correct += 1
            else:
                incorrect += 1

            if abs(out - targ) <= 1:
                within1 += 1
            
            total_sqerr += (out - targ) ** 2
            tot_l1 += abs(out - targ)

        except json.JSONDecodeError:
            print("Bad output:", output)

    rmse = (total_sqerr / (correct + incorrect)) ** 0.5
    l1 = tot_l1 / (correct + incorrect)
    print(
        f"{correct} correct, {incorrect} incorrect ({correct / (correct + incorrect) * 100:.1f}%)  RMSE: {rmse:.2f}  L1: {l1:.2f}  Within1: {within1 / (correct + incorrect) * 100:.1f}%"
    )
    print("Ratings:", ratings)

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
