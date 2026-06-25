# Deep Learning Applications - Labs

Laboratory submissions for the **Deep Learning Applications** course at the **University of Florence (UniFI)**.

The course covers deep learning methods and their practical applications.


## Structure

The repository contains three laboratory folders. Each homework is self-contained.  
All relevant information are conteined in the notebooks (.ipynb)

<pre style="font-family: monospace; font-size: 14px; line-height: 1.2;">
<span style="color:#4285f4"><b>DLA/</b></span>
├── README.md
├── <span style="color:#4285f4"><b>1-Homework/</b></span>
│   ├── DLA-Lab1.ipynb
│   ├── pyproject.toml
│   └── uv.lock
│   
├── <span style="color:#4285f4"><b>2-Homework/</b></span>
│   ├── DLA-Lab2.ipynb
│   ├── pyproject.toml
│   └── uv.lock
│   
└── <span style="color:#4285f4"><b>3-DRL/</b></span>
    ├── DLA-Lab3-DRL.ipynb
    ├── pyproject.toml
    └── uv.lock
</pre>

---

## List of choosen exercizes 

| Lab   | Assignment   |
| ---------- | ------------ |
| Homework 1 | Exercise 3.2 (Training free classification) |
| Homework 2 | Exercise 3.3 (Text-To-Image-App [Try here](https://huggingface.co/spaces/Andy-6/Text-to-ImageApp)) |
| Homework 3 | Exercise 3.3 (OpenaAi-car) |

---

## Setup

Dependencies are managed independently within each homework using `uv`.

Install `uv` if it is not already available:

```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

Then enter the desired homework folder and install its dependencies:

```bash
cd <Lab_folder>
uv sync
```
