# Unsloth LLM Fine Tuner for Equity Ratings

Fine tune phi-4 14b to rate equities as either sell (rating=3), hold (rating==3), or buy (rating>3) based on past financial information (statements, prices, earnings transcripts, analyst ratings, insider trades, and more). Also predict 30d, 60, 90, 180d, and 365d future returns.

Train:

`python main.py train --out lora_model`

Test:

`python main.py test --checkpoint lora_model`

## Data

See `data/example.json` and `data/example.jsonl.zst` for minimal test data.
