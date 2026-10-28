# Benchmark filter report (D-025)

Documents dropped at prep time by the 13-word n-gram filter, per source and per benchmark. Built from `manifests/*.json` on 26 Oct 2026, after the gate 1 prep. Ships with the release (D-010).

## Filter

- Text is lowercased, punctuation is removed, and the text is split on whitespace (`src/slm50m/dataprep/decontam.py`)
- A training document is dropped if it shares any 13-word n-gram with a benchmark item
- Items: `slm50m-data:/benchmarks/items.jsonl`, one line per context + ending

| Benchmark | Split | Items (context + ending) | Items under 13 words |
|---|---|---|---|
| HellaSwag | validation | 40,168 | 0 (0%) |
| ARC-Easy | test | 9,501 | 1,619 (17%) |
| ARC-Challenge | test | 4,687 | 461 (10%) |
| PIQA | validation | 3,676 | 820 (22%) |
| ArithMark-3 | all | 4,000 | 32 (1%) |

An item under 13 words has no 13-gram, so the filter can't match it. That's a known limit of the GPT-3 method, and it mostly affects short ARC and PIQA items.

## Dropped documents

| Source | Files | Docs seen | Dropped | Per million | HellaSwag | ARC-Easy | ARC-Challenge | PIQA | ArithMark |
|---|---|---|---|---|---|---|---|---|---|
| climbmix | 185 | 15,654,912 | 985 | 62.9 | 622 | 212 | 48 | 103 | 0 |
| fineweb_edu | 4 | 3,250,736 | 120 | 36.9 | 77 | 37 | 2 | 4 | 0 |
| finemath | 11 | 1,151,480 | 115 | 99.9 | 31 | 30 | 14 | 40 | 0 |
| txt360_web | 3 | 2,792,447 | 108 | 38.7 | 101 | 2 | 0 | 5 | 0 |
| txt360_qa | 1 | 556,617 | 53 | 95.2 | 42 | 4 | 3 | 4 | 0 |
| txt360_nltk_qa | 1 | 645,016 | 52 | 80.6 | 37 | 8 | 1 | 6 | 0 |
| wikihow | 2 | 179,191 | 22 | 122.8 | 21 | 0 | 1 | 0 | 0 |
| cosmopedia_v2 | 5 | 1,881,445 | 19 | 10.1 | 15 | 3 | 0 | 1 | 0 |
| word_problems | 33 | 2,770,540 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| arith | 20 | 6,145,632 | not filtered | | | | | | |
| **Total (real data)** | 245 | 28,882,384 | 1,474 | 51.0 | 946 | 296 | 69 | 163 | 0 |

- Docs seen for word_problems counts only the OpenMathInstruct-2 rows kept by the GSM8K filter, plus all of Orca
- Arith isn't filtered at prep. The generator is checked instead: zero exact matches and zero 13-gram hits against ArithMark-3 in 54,000 problems, every template and format (D-011, gate 0)

## Reading

- About 1 in 20,000 documents is dropped. The model loses almost nothing
- HellaSwag gives 64% of the hits. It's built from WikiHow and ActivityNet captions, which web pages copy. WikiHow (Cosmopedia's synthetic WikiHow) has the highest rate, 123 per million, as expected
- FineMath (100 per million) is the one source where ARC and PIQA hits outnumber HellaSwag: 84 of its 115. The TxT360 QA sources (81-95 per million) are mostly HellaSwag hits
- ArithMark: zero hits in every source, including the word-problem sets
- Cosmopedia v2 is synthetic and has the lowest rate among filtered sources, 10 per million
