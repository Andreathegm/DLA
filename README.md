# Deep Learning Applications — Laboratory Sessions

Laboratory submissions for the **Deep Learning Applications** course at the **University of Florence (UniFI)**.

The course covers deep learning methods and their practical applications. These labs put into practice the concepts and tools introduced during lectures.

---

## Structure

The repository contains four laboratory folders. Each lab is self-contained with its own code, configuration, and results.

```
DLA/
├── README.md              
├── pyproject.toml          ← all dependencies managed with uv
├── uv.lock
├── 1-Homework/
│   ├── README.md           ← lab description,main results
│   ├── analysis.ipynb      ← final results, plots, and commentary
│   ├── main.py
│   └── src/
├── 2-Homework/
│   ├── README.md
│   ├── analysis.ipynb
│   └── ...
├── 3-Homework/
│   ├── README.md
│   ├── analysis.ipynb
│   └── ...
└── 4-Homework/
    ├── README.md
    ├── analysis.ipynb
    └── ...
```

Each `README.md` describes the specific lab: task, implementation choices, and a summary of results. Each `analysis.ipynb` contains the full result analysis with plots and comments.

---

## Labs

| Lab | Topic | 
|-----|-------|
| [Lab 1](./lab1/README.md) | Retrieval as Training-free Classification on GTSRB |  |
| Lab 2 | TBD | 
| Lab 3 | TBD |
| Lab 4 | TBD |

---

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv). Install it if you don't have it:

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh

```

Then install all dependencies and activate the environment:

```bash
uv sync
```

---

## Usage

Run a lab's main script:

```bash
uv run python lab1/main.py
```

Open a lab's analysis notebook:

```bash
uv run jupyter notebook lab1/analysis.ipynb
```