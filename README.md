# Obsidian Chat

Super quick hack to pull a ton of cruft out of my bare metal implementation to leverage langchain convenience wrappers

Defaults to using the gpt4all "mpt-7b-instruct" model in code. Very baseline, but totally offline and basically
functional.

## Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python3 main.py /path/to/obsidian/vault"
```

## Tuning

Still a bunch of hackable tuning options in the code, still deciding what I want to do with them.

The tunings are very different to accomodate different commercial or open LLMs, so I am thinking maybe some wrappers,
but I don't want to box people in.
