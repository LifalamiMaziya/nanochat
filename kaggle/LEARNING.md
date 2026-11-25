# How a chat model gets made: notes from our own runs

Plain-English notes on each stage of nanochat, using the numbers from our Kaggle pilot
(25 Nov 2026, 1x Tesla T4, a 6-layer model). Updated as we run more.

## The model we trained
- 6 layers, 384 wide, 6 attention heads, vocabulary of 32,768 tokens, 512-token context.
- Tiny: GPT-2 small has 12 layers and 768 wide. `--depth` is the one dial; everything else scales from it.

## Stage by stage (pilot v2 times)

| Stage | Time | What it does |
|---|---|---|
| install | 3 min | Installs PyTorch and the small libraries nanochat needs |
| data | 6 s | Downloads 4 shards of web text (ClimbMix) to learn from |
| tokenizer | 78 s | Learns how to cut text into "tokens" (word pieces). 32,768 pieces, trained on 1 billion characters |
| pre-training | 6 min (4.5 min of actual training) | The model reads 16 million tokens and learns to guess the next token. This is where language comes from |
| base eval | 23 s | Quizzes the raw model on small test sets (ARC, HellaSwag, PIQA...) |
| SFT | 2 min | "Supervised fine-tuning": shows the model example conversations so it learns to act like an assistant |
| chat sample | 15 s | Asks the finished model two questions |

Total: about 13 minutes of the usual 30-hour weekly Kaggle GPU allowance.

## What the numbers mean
- **Loss** is how wrong the next-token guesses are. It started at 10.4 (pure guessing among 32,768 tokens:
  ln(32768) = 10.4) and fell to 4.4 after 1,000 steps. Lower is better.
- **bpb (bits per byte)** is the same idea measured per character of text, so it can be compared across
  tokenizers. Ours: 1.33 on unseen text. Big models get well under 1.0.
- **Speed:** about 58,000 tokens per second on the T4. An 8xH100 node does millions.
- **Eval scores** around 25% on 4-choice tests mean "guessing". Expected at this size; PIQA (69%) and
  CommonsenseQA (50%) show a little real signal.

## What the raw model sounded like after pre-training
Prompt "The capital of France is" -> "a country where the country is a country where the country is..."
It writes grammatical-looking English but knows nothing and loops. This is normal for 16M tokens:
grammar is learned first, facts much later. (This lines up with the cognitive-core idea: structure and
knowledge are separate things the model picks up at different rates.)

## The first real bug: SFT blew up
In SFT the loss went 2.5 -> 3.9 -> 4.5 -> NaN ("not a number") by step 4, and the chat model
then printed junk tokens. A loss that rises and then becomes NaN means training "diverged": the
updates were too big and the numbers overflowed. Two suspects:
1. **Learning rate too high.** SFT restarts at 80% of the peak pre-training rate.
2. **fp16 overflow.** The T4 can't use bf16, so we run fp16, which has a much smaller number range.
Pilot v3 tests each one on its own (fp32 with the normal rate, then fp16 with a 10x lower rate).

## Words
- **Token:** a word piece. "unbelievable" might be "un" + "believ" + "able".
- **Step:** one update of the model's weights after looking at one batch (here 16,384 tokens).
- **Checkpoint:** a saved copy of the model's weights, so training can stop and resume.
- **fp16 / bf16 / fp32:** number formats. Fewer bits = faster and smaller, but less range/precision.
