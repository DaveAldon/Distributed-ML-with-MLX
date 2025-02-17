<p align="center">
  <img src="/resources/banner.png" alt="alt text" width="75%">
</p>

🍎👉🍏 Everything you need in order to get started building distributed machine learning with Apple's MLX. This repo covers the comprehensive fundamentals to fine tune MLX models with multiple computers. It's put together from my blog series on [Distributed Machine Learning Fundamentals](https://www.grandrapidsdeveloper.com/blog/Distributed-Machine-Learning-Fundamentals).

You'll find all of the relevant files referenced in the readme inside this repo:

- `data_parser_and_splitter.py` - A python script that takes a JSONL file and formats it for LoRA, and splits it into training and validation sets
- `logistic_regression.py` - A python script that builds a minimal logistic regression model from scratch using MLX
- `fine_tune.py` - The fine tuning script that runs MPI through MLX and performs gradient averaging on a model
- `data-sets/` - The original and parsed datasets used in this walkthrough
- `adapters/` - Contains the solo and distributed adapters outputted from my own training
- `hostfile` - Example MPI hostfile for running on multiple computers

Let's get started!

### Our use case

Wowed by all of the fancy models being produced every week it seems, and the agency offered by locally hosting them with products like webAI, I wondered how I could get by without a super computer. Luckily, running inference on these advanced models doesn't take much more than a Raspberry Pi in some cases. But what if the model doesn't do what I need it to do? What if I wanted it to do something really niche, teach it something new or not publicly available?

**I always wanted my chatbot model to be able to talk to me like a pirate.**

![](/resources/pirate.png)

If we take a model like `Mistral-7B-v0.3-4bit`, which is a great model to train with when you don't have a lot of RAM, asking it to talk like a pirate is...lackluster:

```bash
mlx_lm.generate --model ../Mistral-7B-Instruct-v0.3-4bit --prompt "Tell me about greek and roman history like a pirate"
==========
Ahoy matey! Let's set sail through the annals of Greek and Roman history, like a ship navigating the vast sea of time!

First, we'll anchor at the shores of ancient Greece, in the cradle of Western civilization. The Greeks, they were a clever bunch, with city-states like Athens and Sparta leading the charge. Athens, known for its wisdom, was home to philosophers like Socrates, Plato...
==========
Prompt: 17 tokens, 85.875 tokens-per-sec
Generation: 100 tokens, 37.294 tokens-per-sec
Peak memory: 4.119 GB
```

It does okay. It knew to say "Ahoy matey!" But really drops the ball on pirate grammar as the sentences continue.

We have two options to fix this:

- RAG (Retrieval-Augmented Generation)
- Fine Tuning

#### RAG - Better for Variety

![](/resources/rag.png)

With a RAG, we can give our model access to a Vector or Graph database that houses random information that it otherwise wouldn't know, like company data, live metrics, or to oversimplify things, whatever suits a google spreadsheet really well.

#### Fine Tuning - Better for Specialization

![](/resources/finetune.png)

Fine tuning essentially means taking an existing model and training it on a new dataset that is more specific to the task you want it to perform. The output of fine tuning might be safetensor files, and you could fuse them to the model to output an entirely new model, or keep them separate as an adapter. This is better suited, in the case of a chat model, for teaching the model a new language, writing style, or commiting new information to the model itself.

### Since we want to teach the model how to speak differently, fine tuning is the way to go

So can we do any of this without a super computer? Absolutely. For fine tuning, we would need a dataset, a model to fine tune, and...enough RAM to support the training process. Oops! Does that mean we need a super computer to fine tune a big model?

### No

![](/resources/oldcomputer.jpg)

Distributed machine learning lets us take an otherwise "too large" model and *distribute* the workload across a bunch of bargain bin computers.

Let's assume you're already really bored by now and just want more bullet points. Here's our agenda for this walkthrough, and what each section will be about:

1. **Introduction** - The bird's eye view, and high level questions and requirements to consider if this is applicable to you
2. **Preparing a Model with MLX** - Setting up MLX and running inference on a model to establish a baseline. Otherwise, how will we know that a distributed workload sped anything up?
3. **Dataset Preparation** - Getting the pirate grammar formatted for fine tuning. Curating data is incredibly integral to machine learning. If you don't get this right, you will fail to get any good results
4. **How to setup MPI** - Everything you need to know to get MPI synchronizing processes on multiple computers. We'll walk through SSH, Hosts, Thunderbolt, and how ChatGPT doesn't have any answers to the problems you'll likely face
5. **Distributed Fine Tuning** - Combine everything we've learned to fine tune a model across multiple computers
6. **Next Steps** - The limitations of our fundamentals, and where to go from here to get things to be more sophisticated

Let's get started.

## Introduction

![](/resources/macs.jpg)

Is distributed machine learning right for you? Consider these questions:

- Is training and fine tuning taking too long for you on your current setup?
- Do you have multiple computers available?
- Do you have a dataset that is too large to fit in memory?
- Are you applying for a job at NVIDIA and need to know more about MPI?

If you're going to follow along closely with the code in this series, you'll need the following:

- 1 (preferrably 2) or more Apple Silicon Macs (We're using MLX which doesn't work on anything else)
- A thunderbolt (preferably thunderbolt 4) cable

If you don't have these things, you'll still leave with fundamental knowledge that you can apply to other tech stacks. The underlying principles are the same.

## The Bird's Eye View

1. When we fine tune a model, we're giving it new information and testing it on that information to see if it's learned anything

2. When we distribute the fine tuning process, we're testing each computer and *averaging* their results, or gradients

You can distribute the work using a lot of methods, but one of the most common is MPI (Message Passing Interface). MPI is one of many standards for synchronizing processes across multiple processors and computers. Instead of a game of telephone, where messages get distorted as they pass along, it's more like a synchronized group chat where all nodes share their updates and gradients in parallel.

The more complex the model, the more complex the weights, and therefore the more bandwidth you need to share the weights across your computers. If you distribute your training across wifi, you'll probably negate the entire benefit of distributing the work in the first place. This is why I prefer thunderbolt 4 which has more than enough bandwidth.

# Preparing a model with MLX

My goal with this section is to get anyone comfortable with running inference on a model using Apple's open source MLX framework, and how to choose a model that is right for you. We'll be doing the following:

- Installing MLX
- Installing a good model
- Running inference using MLX
- Creating a simple logistic regression model with only MLX

## MLX

Verbatim from their [own repo](https://github.com/ml-explore/mlx), "MLX is an array framework for machine learning on Apple silicon, brought to you by Apple machine learning research." It is an open source competitor to industry staples like TensorFlow and PyTorch, and is specifically designed to run on Apple Silicon natively.

I wanted to use MLX for this series not just because it's newer and more topical, but because from my experience it's a *lot* easier to run on Apple Silicon, and it has less documentation which means we have to actually figure things out for ourselves, and build up fundamental knowledge. Let's get started.

## Requirements

MLX has its own [requirements](https://ml-explore.github.io/mlx/build/html/install.html):
- Using an M series chip (Apple silicon)
- Using a native Python >= 3.9
- macOS >= 13.5

Make sure to also have [pyenv](https://github.com/pyenv/pyenv) installed, because being able to manage your python versions for individual projects is a good practice.

## Installing MLX

If you follow MLX's installation guide, you might end up frustrated, so try these steps instead:

- Create a folder and cd to it in terminal
- Create a new python virtual environment (otherwise you'll likely get segmentation fault errors):
```bash
mkdir mlx-project
cd mlx-project
python3 -m venv mlx-env
source mlx-env/bin/activate
```
- Tell `pyenv` to use the python version that you want in this virtual environment (I'm using 3.13.1):
```bash
pyenv local 3.13.1
```
- Verify that the right version is being aliased in this directory:
```bash
python --version
```
- If it's wrong, you may have a `pyenv` installation issue. You can check out their docs on your own, or oftentimes the quick temporary fix is to run this in your current terminal session:
```bash
eval "$(pyenv init --path)"
```
- Then install the MLX packages:
```bash
pip install -U mlx mlx-lm
```
- Make sure you're in your parent directory, and just run `mlx_lm.lora` in the terminal. You should get a bunch of errors, but no errors about the command not being found. This means it's installed and ready to go.

## Getting a Model

The obvious key to distributed fine tuning, is having a model to fine tune. For this series, we'll be working with something that's not super sophisticated, but is also smart enough that we won't have to do a crazy amount of fine tuning to get some results.

I chose [Mistral-7B-Instruct-v0.3-4bit](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit). This is not to be confused with its big brother, [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).

You can use the more sophisticated model if you like, however the smaller one fulfills a couple needs:
- It's small and therefore faster to download, fine-tune, and run inference on
- I have a 32GB RAM mac, and a 16GB RAM mac for my distributed setup. 16 GB simply isn't enough for the larger model, but the 4bit version is perfect. You might be wondering why distributed machine learning doesn't solve this by sharing the load of the larger model across multiple computers. We will tackle this concept later on in the series
- It's a "dumb" enough model that I've consistently gotten the same responses from it over and over, which is actually really good for testing

### How to download the model

- You'll need an account, and you have to accept the terms and conditions to allow your account to download it
- Make a user access token [here](https://huggingface.co/settings/tokens)
- Install the Huggingface cli
```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```
- Enter your token, then run this command to verify:
```bash
huggingface-cli whoami
```
- Next, install `git-lfs` because the model is several gigabytes:
```bash
git lfs install
```
- Clone the model (I put it outside of my project folder):
```bash
git clone https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit
```

### Running inference

To run inference on the model, run this command (make sure to point to the model you downloaded):
```bash
mlx_lm.generate --model ../Mistral-7B-Instruct-v0.3-4bit --prompt "Tell me about greek and roman history like a pirate"
```

You should get a response like this:
```bash
==========
Ahoy matey! Let's set sail through the annals of Greek and Roman history, like a ship navigating the vast sea of time!

First, we'll anchor at the shores of ancient Greece, in the cradle of Western civilization. The Greeks, they were a clever bunch, with city-states like Athens and Sparta leading the charge. Athens, known for its wisdom, was home to philosophers like Socrates, Plato
==========
Prompt: 17 tokens, 88.261 tokens-per-sec
Generation: 100 tokens, 36.407 tokens-per-sec
Peak memory: 4.119 GB
```

If you get a response like this, then everything is working well. If you want to know how to do this programmatically, you can accomplish the same thing with python:
```python
from mlx_lm import load, generate

model, tokenizer = load("../Mistral-7B-Instruct-v0.3-4bit")

prompt = "Tell me about greek and roman history like a pirate"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

## What's going on

This is a really basic example of working with a model in MLX. You'll find a lot of tutorials going over this out there, so I wanted to take this a step further into territory that isn't as documented: starting from scratch.

To build up our fundamentals, let's make a minimal logistic regression model built entirely with MLX arrays and plain Python to learn the following:
- How MLX differs from TensorFlow and PyTorch (if you're familiar with them, you'll notice terminology differences)
- To introduce you to gradients which will be important later in this series
- To show you how to build a model from scratch that you can train in a couple seconds

We're going to create a model that can predict the `OR` function. This is a simple binary function that returns `1` if either of the inputs are `1`, and `0` if both are `0`.

Let's start by setting up the data using MLX arrays:
```python
# The input data, a 2D matrix
X = mx.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# The output data, or "labels"
y = mx.array([0, 1, 1, 1])
```

Next, we initialize the model's parameters. We're dealing with two input features, which is an individual measurable characteristic of the data being used in the model. Features are the inputs to the model that help it to make a prediction.

In our case, the input features are binary values (0 or 1) in the input array.

```python
# For two input features, we need a weight vector of shape (2,) which is a 1D array with 2 elements
w = mx.zeros(2)
# This is a bias term, an additional parameter that allows the model to fit the data better
# by shifting the decision boundary
b = 0.0
# This determines how much the model's parameters (weights and bias) are adjusted during 
# each step of the training process. It determines the size of the steps taken towards 
# minimizing the loss function
learning_rate = 0.1
# The number of complete passes the model makes through the entire dataset during training.
# During an epoch, the model processes each training example once and updates its parameters 
# (weights and biases) based on the computed gradients
num_epochs = 1000
```

The learning rate is interesting because if it's too high, the model may take large steps and overshoot the optimal values of the parameters, leading to divergence or oscillation. If it's too low, the model will take very small steps, resulting in a slow convergence and possibly getting stuck.

Next we can define a couple of helper functions:

```python
# Maps any real number to the range [0, 1]
def sigmoid(z):
  return 1 / (1 + mx.exp(-z))

# Computes the model prediction. 
# We input X as the data
# w as the weights which determine how important each input is
# b for bias to make better guesses
def predict(X, w, b):
  b_array = mx.full((X.shape[0],), b)
  return sigmoid(mx.addmm(b_array, X, w))

# Measures how good or bad the model's predictions are compared to the actual labels
def binary_cross_entropy(y_true, y_pred, eps=1e-8):
  return -mx.mean(
    y_true * mx.log(y_pred + eps) + (1 - y_true) * mx.log(1 - y_pred + eps)
  )
```

Now, we create our training loop:
```python
for epoch in range(num_epochs):
    # Forward pass which computes predictions and loss
    y_pred = predict(X, w, b)
    loss = binary_cross_entropy(y, y_pred)

    # Backwards pass which computes gradients manually. This essentially helps us teach
    # the model how wrong it was in a bad prediction, so that it can learn.
    grad = y_pred - y
    # We look at the effect of each input on the wrong guesses and averages these effects
    grad_w = mx.addmm(mx.zeros_like(X.T @ grad), X.T, grad) / X.shape[0]
    # Calculates how much the bias needs to change. It averages the effect of the bias on the wrong guesses
    grad_b = mx.mean(grad)
    # Update our parameters based on the gradients
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # Print progress every 100 epochs.
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss}")
```

Finally, we can output the predictions:
```python
# If the predicted probability is greater than 0.5, it is classified as 1 (true)
# Otherwise, it is classified as 0 (false)
final_preds = predict(X, w, b) > 0.5
print("Final Predictions:", final_preds)

# Calculate the accuracy of the model
accuracy = mx.mean(final_preds == y)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

Running this script should yield similar results to the following:
```bash
python script.py
Epoch    0 | Loss: 0.6931471824645996
Epoch  100 | Loss: 0.342264860868454
Epoch  200 | Loss: 0.2668907940387726
Epoch  300 | Loss: 0.21739481389522552
Epoch  400 | Loss: 0.18253955245018005
Epoch  500 | Loss: 0.1567975878715515
Epoch  600 | Loss: 0.13707442581653595
Epoch  700 | Loss: 0.1215219795703888
Epoch  800 | Loss: 0.10897234827280045
Epoch  900 | Loss: 0.0986524447798729
Final Predictions: array([False, True, True, True], dtype=bool)
Accuracy: 100.00%
```

Our model has predicted the `OR` function with 100% accuracy, using purely MLX and python.

Pay close attention to our use of "gradients" in this example, because I mentioned "gradient averaging" in the last section as a foundational element to distributed machine learning:

- The model makes a guess at whether our input is an `OR` or not, and moves forward in progress
- The model then compares its guess to the actual output, and calculates how far off it was from the right answer
- The model is then told to move in the opposite direction of the error, so that it can learn from its mistakes
- This process is repeated until the epochs are finished

You'll notice that this training script doesn't include outputting a model file, and I didn't really feel like getting into that because I wanted to keep this as relevant to our overarching distributed topic as much as possible. Gradients are very important to understand for future sections, and this minimal example helps shed a little light into what's going on.

## What's Next

Armed with MLX and some basic models, we can now move on to acquiring and preparing a dataset for fine tuning. The `Mistral-7B-Instruct-v0.3-4bit` model is simply not smart enough to talk like a pirate consistently throughout its entire response, and we need to fix that with a great pirate lingo dataset. But you can't just take any piece of data and feed it into a model. It requires formatting, curation, and validation sets to ensure quality results.

## Further Reading

Check out this [repo](https://github.com/LucasSte/MLX-vs-Pytorch) that goes over benchmarking MLX vs Pytorch.

Here is the full python script for the logistic regression model:

```python
import mlx.core as mx

# The input data, a 2D matrix
X = mx.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# The output data, or "labels"
y = mx.array([0, 1, 1, 1])

# For two input features, we need a weight vector of shape (2,) which is a 1D array with 2 elements
w = mx.zeros(2)
# This is a bias term, an additional parameter that allows the model to fit the data better
# by shifting the decision boundary
b = 0.0
# This determines how much the model's parameters (weights and bias) are adjusted during
# each step of the training process. It determines the size of the steps taken towards
# minimizing the loss function
learning_rate = 0.1
# The number of complete passes the model makes through the entire dataset during training.
# During an epoch, the model processes each training example once and updates its parameters
# (weights and biases) based on the computed gradients
num_epochs = 1000

# Maps any real number to the range [0, 1]
def sigmoid(z):
    return 1 / (1 + mx.exp(-z))

# Computes the model prediction.
# We input X as the data
# w as the weights which determine how important each input is
# b for bias to make better guesses
def predict(X, w, b):
    b_array = mx.full((X.shape[0],), b)
    return sigmoid(mx.addmm(b_array, X, w))

# Measures how good or bad the model's predictions are compared to the actual labels
def binary_cross_entropy(y_true, y_pred, eps=1e-8):
    return -mx.mean(
        y_true * mx.log(y_pred + eps) + (1 - y_true) * mx.log(1 - y_pred + eps)
    )

for epoch in range(num_epochs):
    # Forward pass which computes predictions and loss
    y_pred = predict(X, w, b)
    loss = binary_cross_entropy(y, y_pred)

    # Backwards pass which computes gradients manually. This essentially helps us teach
    # the model how wrong it was in a bad prediction, so that it can learn.
    grad = y_pred - y
    # We look at the effect of each input on the wrong guesses and averages these effects
    grad_w = mx.addmm(mx.zeros_like(X.T @ grad), X.T, grad) / X.shape[0]
    # Calculates how much the bias needs to change. It averages the effect of the bias on the wrong guesses
    grad_b = mx.mean(grad)
    # Update our parameters based on the gradients
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # Print progress every 100 epochs.
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss}")

# If the predicted probability is greater than 0.5, it is classified as 1 (true)
# Otherwise, it is classified as 0 (false)
final_preds = predict(X, w, b) > 0.5
print("Final Predictions:", final_preds)

# Calculate the accuracy of the model
accuracy = mx.mean(final_preds == y)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

# Dataset Preparation

In our current problem, we want our model to talk like a pirate really well, and not just use a couple pirate words at the beginning of its response and call it good.

As evidenced by running inference on the model we downloaded in the last section, we get this result:

```bash
mlx_lm.generate --model ../Mistral-7B-Instruct-v0.3-4bit --prompt "Tell me about greek and roman history like a pirate"

==========
Ahoy matey! Let's set sail through the annals of Greek and Roman history, like a ship navigating the vast sea of time!

First, we'll anchor at the shores of ancient Greece, in the cradle of Western civilization. The Greeks, they were a clever bunch, with city-states like Athens and Sparta leading the charge. Athens, known for its wisdom, was home to philosophers like Socrates, Plato
==========
Prompt: 17 tokens, 88.261 tokens-per-sec
Generation: 100 tokens, 36.407 tokens-per-sec
Peak memory: 4.119 GB
```

In order for the model to get smarter, and to learn a new dialect, we outlined in the introductory section that *fine tuning* is the key. Fine tuning is the process of adding more information to a model. So, in our example, what kind of information do we need to provide? We need examples of pirate grammar, of course! And we need a lot of it. So, how do we do that?

## Going Shopping for Data

The key to getting good training results, is a good dataset. In our case, we're expecting to prompt our model with a question of some kind, and we expect it to give an answer in a certain way. This means that the best type of data to fit this is a lot of question and answer prompts.

The dataset we'll use in this section can be downloaded from [here](https://huggingface.co/datasets/TeeZee/dolly-15k-pirate-speech), or a copy is in this repo inside `data-sets/raw_databricks-dolly-15k-arr.jsonl`. This dataset is a collection of 15,000 question and answer pairs, all in pirate speak, and is organized like this:

| Instruction                          | Context                                                                          | Response                                                                                                                                                                                                                                                                                                                                                                                                                                | Category      |
| ------------------------------------ | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Please summarize what Linkedin does. | LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform | Linkedin be a social platform that business professionals create profiles on and network with other business professionals. It be used to network, career development, and for jobseekers to find jobs. Linkedin has over 900 million users from over 200 countries. Linkedin can be used to post resumes/cvs, organizing events, joining groups, writing articles, publishing job postings, posting picture, posting videos, and more! | summarization |

So, does this mean we can just start fine tuning? No. We need to check our model's [documentation](https://docs.mistral.ai/capabilities/finetuning/) first. This is where the practice of reading comes in handy:

![](/resources/docs.png)

We can see [here](https://huggingface.co/datasets/TeeZee/dolly-15k-pirate-speech/blob/main/databricks-dolly-15k-arr.jsonl) that our dataset is actually already in the `.jsonl` format. 

In light of this, you might be tempted to run this simple command to begin fine tuning:

```bash
mlx_lm.lora --train --model ../Mistral-7B-Instruct-v0.3-4bit --data databricks-dolly-15k-arr.jsonl
```

This is a pretty common method of training. LoRA (Low-Rank Adaptation) is a lightweight method of training that helps us adjust large models to new contexts, which we need for fine tuning to avoid having to retrain the entire model on its original dataset. You can read more about it [here](https://huggingface.co/docs/diffusers/en/training/lora).

The command above is great and easy, but it won't work, and the error is interesting:

```bash
raise ValueError(
        "Training set not found or empty. Must provide training set for fine-tuning."
    )
ValueError: Training set not found or empty. Must provide training set for fine-tuning.
```

What does that mean? The training data is right there! Well, actually the problem is that we need to point it to a directory instead, and call it `train.jsonl`. So we do that and try again and get:

```bash
ValueError: Validation set not found or empty. Must provide validation set for fine-tuning.
```

Ok, so what's a validation set? Why doesn't the pirate dataset already come with this?

## Training and Validation

When a model is trained on a bunch of data, it's important to know how well it's doing. This is where the validation set comes in. The validation set is a subset of the training data that the model doesn't see during training. This is important because if the model sees the same data during training and validation, it can memorize the data and not actually learn anything. This is called **overfitting**, which is an important terminology to know for model training and fine tuning.

In the real world, we might liken this to rote memorization. If you memorize a bunch of facts for a test, you might do well on only the questions you effectively memorized. But if you're asked a question that's similar to what you memorized, but not exactly the same, you'll get it wrong.

So how do we take the data and divide it up 80/20? This is actually a great task for your favorite AI tool like ChatGPT, which can spit out a python script that does exactly what we need:

```python
import json
import random
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Split a JSONL file into train and valid sets.')
parser.add_argument('input_file', type=str, help='Path to the input JSONL file')
parser.add_argument('train_file', type=str, help='Path to the output train JSONL file')
parser.add_argument('valid_file', type=str, help='Path to the output valid JSONL file')
args = parser.parse_args()

# Read the input file
with open(args.input_file, 'r') as f:
    lines = f.readlines()

# Shuffle the lines
random.shuffle(lines)

# Calculate the split index
split_index = int(0.8 * len(lines))

# Split the lines into train and valid sets
train_lines = lines[:split_index]
valid_lines = lines[split_index:]

# Write the train lines to train.jsonl
with open(args.train_file, 'w') as f:
    for line in train_lines:
        f.write(line)

# Write the valid lines to valid.jsonl
with open(args.valid_file, 'w') as f:
    for line in valid_lines:
        f.write(line)

print(f"Split {len(lines)} lines into {len(train_lines)} train and {len(valid_lines)} valid lines.")
```

So we run this with our data like so:

```bash
python split.py databricks-dolly-15k-arr.jsonl train.jsonl valid.jsonl

Split 15011 lines into 12008 train and 3003 valid lines.
```

Then we take our two new `.jsonl` files and put them into a `data` folder, and retry our training command:

```bash
mlx_lm.lora --train --model ../Mistral-7B-Instruct-v0.3-4bit --data data
```

And...we get another error!

```bash
ValueError: Unsupported data format, check the supported formats here:
https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data.
```
The link mentioned in the error message tells us that the pirate dataset isn't quite structured correctly for LoRA to use. So, we need to convert it to the correct format as well in our python script.

It's currently structured like this:

```json
{"instruction": "(stuff)", "context": "(stuff)", "response": "(stuff)", "category": "closed_qa"}
```

And it needs to be like this for MLX:

```json
{"messages": [{"role": "user", "content": "(stuff)"}, {"role": "assistant", "content": "(stuff)"}]}
```

So we tweak our script like so and try again:

```python
import json
import random
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Split a JSONL file into train and valid sets.')
parser.add_argument('input_file', type=str, help='Path to the input JSONL file')
parser.add_argument('train_file', type=str, help='Path to the output train JSONL file')
parser.add_argument('valid_file', type=str, help='Path to the output valid JSONL file')
args = parser.parse_args()

# Read the input file
with open(args.input_file, 'r') as f:
    lines = f.readlines()

# Shuffle the lines
random.shuffle(lines)

# Calculate the split index
split_index = int(0.8 * len(lines))

# Split the lines into train and valid sets
train_lines = lines[:split_index]
valid_lines = lines[split_index:]

def format_line(line):
    data = json.loads(line)
    formatted_data = {
        "messages": [
            {
                "role": "user",
                "content": f"You are a helpful assistant.\n\n{data['instruction']}\n\nContext: {data['context']}"
            },
            {
                "role": "assistant",
                "content": data['response']
            }
        ]
    }
    return json.dumps(formatted_data)

# Write the train lines to train.jsonl
with open(args.train_file, 'w') as f:
    for line in train_lines:
        f.write(format_line(line) + '\n')

# Write the valid lines to valid.jsonl
with open(args.valid_file, 'w') as f:
    for line in valid_lines:
        f.write(format_line(line) + '\n')

print(f"Split {len(lines)} lines into {len(train_lines)} train and {len(valid_lines)} valid lines.")
```

Finally, rerunning the training command we get this output:

```bash
Loading pretrained model
Loading datasets
Training
Trainable parameters: 0.047% (3.408M/7248.024M)
Starting training..., iters: 1000
```

Feel free to cancel it if you want, but now we know that the data is good to go.

## The Takeaway

Shopping around can be fun when you're looking for cool datasets for your model to learn, but be prepared to put in a little work to get it formatted properly. There are other considerations we can make as well to take this a step further:

- Read the specs on the model you want to fine tune or train
- Don't forget to split your data into training and validation sets, a good rule of thumb is an 80/20 ratio
- Make sure the data follows the same prompt patterns you expect to use, for best results
- Use AI chat models to spit out some quick python scripts that parse through things, it's really easy
- If memory usage needs to be kept at a minimum, for the Mistral v0.3 model family, remove data that's longer than 2048 tokens or you'll get warnings during the training loops, and your RAM usage will spike, and the training will take longer
- Don't truncate data, it's better to remove it entirely, because it reduces the quality of the training data and the model will learn incorrect patterns

## Next Steps

If all you cared about was fine tuning an MLX model, you could stop here in the walkthrough. You could, with one computer, fine tune whatever you want now with the fundamentals we've covered so far. And that's okay, because that's how distributed machine learning starts: with one machine. But it ends with two or more. We want to take it a step further and get multiple machines to run through the training together to greatly speed up the process.

In the next section, we'll learn how to start synchronizing multiple machines using MPI. After that, we'll combine all of the concepts we've learned and fine tune our model with our distributed setup.

# How to Setup MPI

By now we know how to setup our model, prepare our dataset, and even start a simple fine tuning process. But if we ran all of this on two computers, we would have two separate processes and the output would be different. In this section, we will be discussing how to use MPI to synchronize multiple computers so that the following can happen:

- All computers are connected together
- All computers begin the training process at the same time
- MLX can share gradients across the MPI "world"

## What is MPI?

MPI stands for Message Passing Interface. It is a standard that defines how multiple processes can communicate with each other.

It is commonly used in High Performance Computing (HPC) to synchronize multiple computers, and also to run processes on a GPU in parallel on the same computer. MPI can do a lot of interesting synchronizing tasks in the computing world. In our case, we only need MPI in order to share gradients so that we can average them and benefit from the increased computing power of multiple machines. So the multi computer synchronization feature is what we'll explore.

## Requirements

Let's assume a two computer setup to keep things simple (host and client).

- Install [Open MPI](https://www.open-mpi.org/) on each computer: `brew install open-mpi`
- Setup passwordless SSH between the host and client. This is because MPI needs to be able to invoke commands on the client machine directly:
  - On the client machine run: `ssh-keygen -t rsa` and follow the prompts. For simplicity, you can just press enter for all of them which will default to no password
  - Copy the public key outputted to the host machine: `ssh-copy-id -i <path_to_rsa.pub_file> user@host`, where `user` and `host` is the username and IP address of the host machine
  - Verify it worked by connecting to the client machine from the host without needing to enter a password: `ssh user@host`

## Running MPI

MPI needs a host file in order to know about all of the computers in our distributed setup, and their (ideally private) IP addresses. In the root of your project, create a file called `hostfile` (you can call it whatever you want, even without an extension), and add your information like below:

```text
10.0.0.2 slots=1
10.0.0.3 slots=1
```

The slots parameter is for MPI to know how many processes to run on each machine. In this case, we are running one process on each machine. You can disable a client machine by changing the slots parameter to 0.

> You may find documentation that says that you can use hostnames instead of IP addresses. From my attempts, I've never gotten that to work and have always had to use IP addresses.

You might be wondering why we're just using the network to do this instead of a thunderbolt cable like I mentioned in our intro. And we'll get to that, but first we want to make sure the simpler setup is working.

## Trying it out

To test that our setup is working, run this command on the host machine:

```bash
mpirun --hostfile hostfile -np 2 hostname
```

- The `-np` parameter is the number of processes to run. This needs to match the number of available slots in the hostfile. For example, if your hostfile is setup like earlier, but `-np` is set to 3, you'll get an error
- The final argument is the direct command to run on each machine. In this case, we're just running `hostname` to print out the name of the machine

If this works, you'll get something like this outputted:

```bash
m1-pro-mini.local
m1-max-mini.local
```

Expect this command to resolve quickly. If it hangs, then there's a couple things you can do to diagnose the issue:
- Check that the hostfile is correct
- Check that the host and client machines can communicate with each other by pinging each other's IP addresses
- Check that the client machine can be accessed via SSH without a password on the host machine
- Check that MPI is installed correctly on each machine by running this command separately on each machine: `mpirun -np 1 hostname` (this runs a single process directly on the machine without needing to resolve hosts)

## Trying more complicated commands

If the above all works, then you're ready to move on. With MPI, we can define any command that gets run on both machines. So, how do we run something with python? We have to be able to run our scripts with this in mind:

- We need to reference the right directories
- We need to use our virtual environment
- We need to be able to pass arbitrary arguments to our script

We can accomplish this with bash strings and relative paths that are the same on both machines:

```bash
mpirun --hostfile hostfile -np 2 bash -c '$HOME/Desktop/Fine-Tuning-Project/MLX-env/bin/python $HOME/Desktop/Fine-Tuning-Project/script.py'
```

This command invokes python directly from the virtual environment on each machine, and let's us pass whatever arguments we want inside the bash string. Try running one of your python scripts with this method before moving on.

## Thunderbolt

Previously, we were running MPI over the default network interface. Now, we need to get MPI working over Thunderbolt, which takes a slight change in commands. Follow these steps:

- Connect the two computers with a Thunderbolt cable
- Under `Settings > Network > Thunderbolt` make sure that each computer is connected and assigned an IP address (ideally on the `bridge0` interface)
- I recommend setting the IP addresses manually if they're not, to something easy to remember
- Ping each computer using their Thunderbolt IP addresses to make sure they can communicate. Another cool way to try this is by using the `Screen Sharing` app, which will be really fast over a Thunderbolt connection
- Update your `hostfile` to use the new sets of Thunderbolt IP addresses

Next, we have to change our MPI command to include some additional parameters:

```bash
mpirun --hostfile hostfile -np 2 \
  --mca oob_tcp_if_include bridge0 \
  --mca btl_tcp_if_include bridge0 \
  bash -c '$HOME/Desktop/Fine-Tuning-Project/MLX-env/bin/python $HOME/Desktop/Fine-Tuning-Project/script.py'
```

> **oob_tcp_if_include** is used to specify which network interfaces should be used for out-of-band (OOB) communication. MPI defaults to the normal `eth0` if this isn't set, so we bind it to `bridge0` to use Thunderbolt

> **btl_tcp_if_include** specifies which network interfaces should be used for the byte transport layer (BTL), which is responsible for actual MPI message passing over TCP. We have to bind this as well from the default `eth0` to `bridge0`

If your Thunderbolt interface isn't bound to `bridge0`, you can find what it's called by running this command:

```bash
ifconfig
```

Then, look through the output to find the IP address you assigned it to, and if it's active (here's an example of `bridge0` if it's not set):

```bash
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	options=63<RXCSUM,TXCSUM,TSO4,TSO6>
	ether 36:60:64:32:40:00 
	Configuration:
		id 0:0:0:0:0:0 priority 0 hellotime 0 fwddelay 0
		maxage 0 holdcnt 0 proto stp maxaddr 100 timeout 1200
		root id 0:0:0:0:0:0 priority 0 ifcost 0 port 0
		ipfilter disabled flags 0x0
	member: en2 flags=3<LEARNING,DISCOVER>
    ifmaxaddr 0 port 9 priority 0 path cost 0
	member: en3 flags=3<LEARNING,DISCOVER>
    ifmaxaddr 0 port 10 priority 0 path cost 0
	nd6 options=201<PERFORMNUD,DAD>
	media: <unknown type>
	status: inactive
```

## What's Next

If running MPI commands over Thunderbolt is working for you, then you're ready for the next section in this walkthrough coming up, which will combine every concept we've learned to finally fine tune our model across multiple computers, and average the gradients over MPI.

## Further Reading

- If you want to start reading more documentation on your own with what you can do with MPI and MLX, they have this documented [here](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- You can also read about how PyTorch uses MPI for distributed work as well [here](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- MPI isn't just for synchronizing computers, but processes and GPUs. NVIDIA uses MPI to solve a lot of problems with CUDA [here](https://developer.nvidia.com/mpi-solutions-gpus)

# Distributed Fine Tuning

It's time to finally combine every fundamental we've covered in this walkthrough and accomplish the full fine tuning process on multiple computers.

By now, you should have the following working:
- A model that's compatible with MLX (ideally `Mistral-7B-Instruct-v0.3-4bit`)
- A dataset compatible with your model
- MPI installed and working across multiple computers

We've alluded to needing gradient averaging in order to get everything to actually work, and that's what this section is all about.

## Gradient Averaging

For our purposes, to understand the fundamental concept of averaging gradient output from multiple computers, we can think of it as a way to combine the results of multiple models into one. We can accomplish this with a very simple python function:

```python
def all_reduce_grads(grads):
    return tree_map(lambda g: mx.distributed.all_sum(g) / size, grads)
```

> `lambda g: mx.distributed.all_sum(g) / size` is an anonymous function (lambda) that takes a gradient g and performs two operations: `mx.distributed.all_sum(g)` sums the gradient g across all MPI ranks, and `size` which is the total number of MPI ranks. This effectively computes the average of the gradient across all ranks.

The "rank" terminology is used in MPI to refer to the unique identifier assigned to each process in a distributed computing environment. In your hosts file, each slot is a rank.

The reason we need this function is because we can't do this strictly through the command line interface into MLX. We need a custom python script that uses their API.

## Putting the Script Together

We effectively need a script that recreates the `mlx_lm.lora` training commands, but add the gradient averaging function as a callback. Let's walk through how to do this:

```python
import argparse
import time
import types
import matplotlib.pyplot as plt # <-- this is for producing a graph that is helpful for visualizing our training accuracy
import datetime
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.lora import run

# This is how we define the "world" of our distributed training. MLX needs to know that we're using MPI, and it can figure out the rest
world = mx.distributed.init()
size = world.size()
```

Next, we define our callbacks:

```python
def all_reduce_grads(grads):
    # I added this check so that we can easily run this script as a single process. Size is always 1 if we only have one slot, or aren't using MPI
    if size == 1:
        return grads
    # Sum across all ranks, then divide
    return tree_map(lambda g: mx.distributed.all_sum(g) / size, grads)

# We need this to extend the TrainingCallback class in order to add our custom gradient averaging function
class MetricsCallback(TrainingCallback):

    def __init__(self):
        # Initialize lists for loss tracking
        self.train_losses = []
        self.val_losses = []

    # This runs after backwards pass but before optimizer step
    def on_after_backward(self, model, grads, step):
        new_grads = all_reduce_grads(grads)
        return new_grads

    # This runs when the trainer reports training loss
    def on_train_loss_report(self, info):
        iteration = info.get("iteration")
        train_loss = info.get("train_loss")
        if iteration is not None and train_loss is not None:
            self.train_losses.append((iteration, train_loss))
            print(f"[Train] Iteration {iteration}: Loss = {train_loss:.4f}")

    # This runs when the trainer reports validation loss
    def on_val_loss_report(self, info):
        iteration = info.get("iteration")
        val_loss = info.get("val_loss")
        if iteration is not None and val_loss is not None:
            self.val_losses.append((iteration, val_loss))
            print(f"[Valid] Iteration {iteration}: Loss = {val_loss:.4f}")
```

A good way to visually see how our training is going is to plot the loss values over time. This will be helpful to compare a single computer running the fine tuning vs. our distributed setup. Ideally, there will be no difference, but the distributed setup will take significantly less time.

```python
def plot_metrics(metrics_callback, save_path=None):
    if not metrics_callback.train_losses and not metrics_callback.val_losses:
        print("No metrics to plot.")
        return

    plt.figure(figsize=(8, 5))
    if metrics_callback.train_losses:
        train_its, train_vals = zip(*metrics_callback.train_losses)
        plt.plot(train_its, train_vals, '-o', label='Train Loss')
    if metrics_callback.val_losses:
        val_its, val_vals = zip(*metrics_callback.val_losses)
        plt.plot(val_its, val_vals, '-o', label='Validation Loss')

    plt.title("Training and Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
```

Finally, we add our main entry point which is mostly boilerplate parameter setup to mimic what we can do with the MLX CLI.

> The most important part here is adding our gradient averaging callback.

```python
def main():
    # Print whether single or distributed
    if size == 1:
        print("Single process mode: no gradient averaging needed.")
    else:
        print(f"Distributed mode: Rank {
              world.rank()} - averaging gradients across {size} ranks.")

    parser = argparse.ArgumentParser(
        description="Run fine-tuning with MLX LM + LoRA.")
    parser.add_argument("--model", type=str, default="../Mistral-7B-Instruct-v0.3-4bit",
                        help="Path or name of the base model.")
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--data", type=str, default="data1/")
    parser.add_argument("--fine-tune-type", type=str, default="lora")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--resume-adapter-file", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default="adapters")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-batches", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-parameters", type=dict,
                        default={"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0})
    parser.add_argument("--lr-schedule", type=str, default=None)

    args = parser.parse_args()

    start_time = time.time()

    # Load the model using the --model parameter
    model = load(args.model)

    # Create the callback that does both:
    #  distributed gradient averaging
    #  metrics logging
    metrics_callback = MetricsCallback()

    # Run the LoRA fine-tuning
    # Orchestrates the training loop and calls callback hooks for training/validation loss, backward pass, etc.
    run(types.SimpleNamespace(**vars(args)),
        training_callback=metrics_callback)

    # Plot the collected metrics
    metrics_name = f"graphs/metrics_{
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_metrics(metrics_callback, save_path=metrics_name)

    end_time = time.time()
    print(f"Script execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
```

## Running the script

With this script put together, I recommend running it on one computer first to make sure it's working and trainable. We have a lot of parameters available, and these have worked best for me to get some quick results:

```bash
python script.py --data data --batch-size 2 --num-layers 8 --iters 100
```

You won't get amazing results with this since the `iters` should be about 1000, but this should run fast and produce adapter files, and a graph of the model's training accuracy. A good sign that it's working is you'll see this output:

```bash
Single process mode: no gradient averaging needed.
Loading pretrained model
Loading datasets
Training
Trainable parameters: 0.047% (3.408M/7248.024M)
Starting training..., iters: 100
```

After it finishes the fine tuning, you should have a graph in your folder that looks something like this:

![](/resources/training-100.png)

This is a common training and validation loss graph. Both values should be decreasing over time. The graph compares the training loss (blue) and the validation loss (orange). The x-axis is the iteration (epoch) number, and the y-axis is the loss value.

> If the validation loss is increasing, you're **overfitting**.

> If the training loss is increasing, you're **underfitting**.

Next, sit back for a while and run it with the full 1000 iterations. We want the graph from this output in order to compare with our distributed output later.

```bash
python script.py --data data --batch-size 2 --num-layers 8 --iters 1000
```

This should output something like this:

![](/resources/training-1000.png)

> The graph should be generally trending downwards, and reaching a point where it's not decreasing much anymore. This is a good sign that the model has been trained well, and we don't need to introduce more iterations.

## Hooking up all our computers

Now that your fine tuning has successfully completed on a single computer, it's time to use MPI and get all our computers helping out. Let's run our script through MPI:

```bash
mpirun --hostfile hostfile -np 2 \
  --mca oob_tcp_if_include bridge0 \
  --mca btl_tcp_if_include bridge0 \
  bash -c '$HOME/Desktop/Fine-Tuning-Project/MLX-env/bin/python $HOME/Desktop/Fine-Tuning-Project/script.py --data data --batch-size 2 --num-layers 8 --iters 1000'
```

> If you have RAM issues, you can reduce the batch size to 1. This will make the training take longer, but it will use less memory. We'll discuss this delicate balance more at the end of the walkthrough.

Upon starting this command, you should see output from both computers. A good test as well is monitoring the memory usage in Activity Monitor:

![](/resources/energy.gif)

> The yellow spike in memory pressure is when the first iteration was completed.

Once everything has completed after a while, you'll have a couple artifacts to look at.

## Safetensors

You should have `.safetensors` files in your `adapters` folder. These are the adapter files that were created during the fine tuning process. We use these to in conjunction with the base model to generate inference with new data.

![](/resources/adapters.png)

If you look inside `adapter_config.json`, it contains all of the parameters used to generate the adapters. This is useful for reproducing the results later, and is like metadata for adapters.

The rest of the files serve as checkpoints during the training process. If training was interrupted on a fine tune that could take several days, you'd want to minimize time lost and start where you left off. Because of this, the most important file to keep is the one with the highest iteration number: `0001000_adapters.safetensors`.

> If you've had trouble creating adapters, inside `adapters/` in this repo are a couple that I made with 100 epochs.

## New Graph

You'll have a new loss and validation graph to look at. Below is mine which was produced by a 32GB RAM M1 Pro, and a 16GB RAM M1 Pro:

![](/resources/training-distributed.png)

Compare that with my graph produced by just the 32GB RAM M1 Pro:

![](/resources/training-1000.png)

They are very similar, which means that the accuracy of our model was not negatively impacted by the distributed fine tuning.

But what about the time impact? With my script, it's always outputting how long everything takes. Here were my results:

| Configuration                     | Time (seconds) |
| --------------------------------- | -------------- |
| 32GB RAM M1 Pro                   | 4259.40        |
| 32GB RAM M1 Pro & 16GB RAM M1 Pro | 2610.67        |

This is an order of magnitude faster (38.7%), without any fancy optimizations, and using just fundamentals.

## The fun part

Now that our fine tuning is done and we have our adapter, how do we know that it works? How do we know that our model can speak like a pirate properly as a result of our **2610.67** seconds of labor?

With our new adapters, run the following in terminal as you should have in previous sections, and keep track of the response:

```bash
mlx_lm.generate --model ../Mistral-7B-Instruct-v0.3-4bit --prompt "Tell me about greek and roman history like a pirate"

==========
Ahoy matey! Let's set sail through the annals of Greek and Roman history, like a ship navigating the vast sea of time!

First, we'll anchor at the shores of ancient Greece, in the cradle of Western civilization. The Greeks, they were a clever bunch, with city-states like Athens and Sparta leading the charge. Athens, known for its wisdom, was home to philosophers like Socrates, Plato
==========
Prompt: 17 tokens, 70.533 tokens-per-sec
Generation: 100 tokens, 35.609 tokens-per-sec
Peak memory: 4.119 GB
```

> Ahoy matey! Let's set sail through the annals of Greek and Roman history, like a ship navigating the vast sea of time! First, we'll anchor at the shores of ancient Greece, in the cradle of Western civilization. The Greeks, they were a clever bunch, with city-states like Athens and Sparta leading the charge. Athens, known for its wisdom, was home to philosophers like Socrates, Plato ...

This is our baseline, disappointing result. Now, provide the adapter we made and run the same inference:

```bash
mlx_lm.generate --model ../Mistral-7B-Instruct-v0.3-4bit --adapter-path adapters --prompt "Tell me about greek and roman history like a pirate"

==========
Arr matey! Greek and roman history be th' foundation of western civilization. Th' greeks be th' first civilization to have a written language and th' first to have a democracy. Th' romans be th' first civilization to have a written language and th' first to have a republic. Th' greeks be th' first civilization to have a written language and th' first to have a democracy. Th' romans be th' first civilization to have a written
==========
Prompt: 17 tokens, 74.425 tokens-per-sec
Generation: 100 tokens, 27.879 tokens-per-sec
Peak memory: 4.132 GB
```

> Arr matey! Greek and roman history be th' foundation of western civilization. Th' greeks be th' first civilization to have a written language and th' first to have a democracy. Th' romans be th' first civilization to have a written language and th' first to have a republic. Th' greeks be th' first civilization to have a written language and th' first to have a democracy. Th' romans be th' first civilization to have a written ...

### WOW! 🎉

Our model is now speaking like a pirate throughout the sentences consistently! Have we made the model dumber with this? Maybe. But at least it's using the grammar we want. If we wanted the outputted information to be better, it takes more data curation as we used a relatively small dataset, so you cannot expect perfection.

In the end, behind the scenes we've taken an ordinary fine tuning process, and applied gradient averaging in order to cut the training time down by **38.7%**! This is the power of distributed machine learning.

## What's next?

While this section wraps up the application of our fundamentals, there are some questions and concerns to address going forward, and some recommendations if you need distributed machine learning for a real world application. What you've worked on in this walkthrough is a very basic implementation, and has a lot of inefficiencies that have to be addressed in a production environment. We will go over this in our final section in this walkthrough.

My goal for the final section is for you to be well equipped with the terminology and general frameworks and tech out there to apply distributed machine learning to your next product.

## Full Script

Here is the full python script we used in this section:

```python
import argparse
import time
import types
import matplotlib.pyplot as plt # <-- this is for producing a graph that is helpful for visualizing our training accuracy
import datetime
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.lora import run

# This is how we define the "world" of our distributed training. MLX needs to know that we're using MPI, and it can figure out the rest
world = mx.distributed.init()
size = world.size()

def all_reduce_grads(grads):
    # I added this check so that we can easily run this script as a single process. Size is always 1 if we only have one slot, or aren't using MPI
    if size == 1:
        return grads
    # Sum across all ranks, then divide
    return tree_map(lambda g: mx.distributed.all_sum(g) / size, grads)

# We need this to extend the TrainingCallback class in order to add our custom gradient averaging function
class MetricsCallback(TrainingCallback):

    def __init__(self):
        # Initialize lists for loss tracking
        self.train_losses = []
        self.val_losses = []

    # This runs after backwards pass but before optimizer step
    def on_after_backward(self, model, grads, step):
        new_grads = all_reduce_grads(grads)
        return new_grads

    # This runs when the trainer reports training loss
    def on_train_loss_report(self, info):
        iteration = info.get("iteration")
        train_loss = info.get("train_loss")
        if iteration is not None and train_loss is not None:
            self.train_losses.append((iteration, train_loss))
            print(f"[Train] Iteration {iteration}: Loss = {train_loss:.4f}")

    # This runs when the trainer reports validation loss
    def on_val_loss_report(self, info):
        iteration = info.get("iteration")
        val_loss = info.get("val_loss")
        if iteration is not None and val_loss is not None:
            self.val_losses.append((iteration, val_loss))
            print(f"[Valid] Iteration {iteration}: Loss = {val_loss:.4f}")

def plot_metrics(metrics_callback, save_path=None):
    if not metrics_callback.train_losses and not metrics_callback.val_losses:
        print("No metrics to plot.")
        return

    plt.figure(figsize=(8, 5))
    if metrics_callback.train_losses:
        train_its, train_vals = zip(*metrics_callback.train_losses)
        plt.plot(train_its, train_vals, '-o', label='Train Loss')
    if metrics_callback.val_losses:
        val_its, val_vals = zip(*metrics_callback.val_losses)
        plt.plot(val_its, val_vals, '-o', label='Validation Loss')

    plt.title("Training and Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    # Print whether single or distributed
    if size == 1:
        print("Single process mode: no gradient averaging needed.")
    else:
        print(f"Distributed mode: Rank {
              world.rank()} - averaging gradients across {size} ranks.")

    parser = argparse.ArgumentParser(
        description="Run fine-tuning with MLX LM + LoRA.")
    parser.add_argument("--model", type=str, default="../Mistral-7B-Instruct-v0.3-4bit",
                        help="Path or name of the base model.")
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--data", type=str, default="data1/")
    parser.add_argument("--fine-tune-type", type=str, default="lora")
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--resume-adapter-file", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default="adapters")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-batches", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-parameters", type=dict,
                        default={"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0})
    parser.add_argument("--lr-schedule", type=str, default=None)

    args = parser.parse_args()

    start_time = time.time()

    # Load the model using the --model parameter
    model = load(args.model)

    # Create the callback that does both:
    #  distributed gradient averaging
    #  metrics logging
    metrics_callback = MetricsCallback()

    # Run the LoRA fine-tuning
    # Orchestrates the training loop and calls callback hooks for training/validation loss, backward pass, etc.
    run(types.SimpleNamespace(**vars(args)),
        training_callback=metrics_callback)

    # Plot the collected metrics
    metrics_name = f"graphs/metrics_{
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_metrics(metrics_callback, save_path=metrics_name)

    end_time = time.time()
    print(f"Script execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
```

# Conclusion & Final Considerations

For my final section in this walkthrough, it's time to go over some of the common sense for applying distributed machine learning to production products, and answer some good questions. This section won't be as technical as the others, but instead will serve as a resource to help you decide what the best tools are to use, and where to go with your new distributed fundamentals.

## Making Decisions

If you're interested in adding a distributed setup to your project, I made a simple flowchart to help determine the best path.

![](/resources/distributed-decision.svg)

Let's go over the questions posed and the recommendations:

### Do you need to do this multiple times?

If your problem only requires you to fine tune or train a model once, and likely not again, then a distributed setup may end up taking longer to implement than just doing the training on a single computer. A narrow or simplified use case doesn't really need a complex setup. I would only recommend distributed for a one time use case, if you're doing it to either learn or prove out a concept for convincing parties to use distributed to save time.

### Does training or fine tuning take 1+ hours?

If your training or fine tuning already doesn't take a lot of time to finish, then the time save won't be very noticeable. I would only recommend using distributed if you're going to see some significant real-world time savings, on a factor of hours or more. Otherwise, the effort to implement it outweighs the small gains you might see.

### The basic, fully parallel distributed setup

This solution is essentially the one presented in this walkthrough. It has its own set of limitations and shouldn't be used as a hammer to every distributed problem. Consider this:

- This solution requires a fully parallel setup across the network
  - Each computer needs its own copy of the model, training data, file paths, etc.
  - The network is not resilient to failure, and if one computer goes down it causes disruptions
  - MPI needs all the machines to be immediately accessible to begin the process, it's not resilient to modular changes

- Amdahl's law

![](/resources/amdahl.png)

You can't just add 1000 computers to your setup and expect a 1 second training cycle. Amdahl's law tells us that the speedup from parallelization is limited by the **sequential portion** of the code. If too much computation depends on serialized execution, adding more nodes will have diminishing returns.

- MPI limitations

MPI itself doesn't optimize for deep learning workloads as well as [Horovod](https://github.com/horovod/horovod) or [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), which are designed specifically for distributed training. With that in mind, why did we use MPI in this walkthrough? I chose it because it's the *most foundational distributed framework* that is the underlying technology for a lot of stuff out there. Understanding MPI gives you the best foundational knowledge to work with more specialized tools, which all use the same principles.

> #### Rule of Thumb: The signs of outgrowing the basic setup
> If you don't fit into my flowchart, here are some more considerations to think about. If you find yourself having to use several of these techniques to optimize your basic setup, it's time to move onto a more advanced framework:
> - Optimizing batch sizes to reduce synchronization costs
> - Using gradient compression techniques to reduce communication overhead
> - Using hierarchical communication (e.g., ring-allreduce in Horovod) to improve efficiency
> - Using high-speed interconnects (like NVLink or Infiniband) for better network performance
> - Choosing asynchronous training where possible to minimize blocking operations

### Sharding & A Mature Distributed framework

Considering the limitations of the basic setup, what can we do? What do we do when we need a scalable, advanced solution? This is where we introduce a new term: *sharding*.

#### **Sharding**

This is the technique of dividing a large machine learning model into smaller, more manageable parts (shards) that can be distributed across multiple devices.

There are a couple different types of sharding:

- Tensor (Layer-Wise) Sharding
  - Different layers of the model are distributed across multiple devices
  - Useful for deep networks where computation at each layer can be parallelized
- Operator (Pipeline) Sharding
  - Splits computations (operators) across devices but keeps the model structure intact
  - Common in transformer-based models like GPT
- Parameter (Weight) Sharding
  - Splits model parameters (weights) across devices
  - Reduces memory consumption per device while maintaining full model computation
- Expert (Mixture of Experts - MoE) Sharding
  - Different experts (sub-models) are placed on separate devices
  - Only a subset of the model is activated per inference, reducing computational load

As you might be feeling from reading this list, sharding is complicated to say the least. It's not something you can just throw together with MPI and expect it to work. Typically, if this is needed, it's best to find a framework that takes care of it for you.

For MLX specifically, [this repo](https://github.com/mzbac/mlx_sharding) is a great resource to get started with sharding.

#### Mature Frameworks

I mentioned a couple tools previously, but there's a spectrum of involvement you'll need. At the time of writing this, MLX isn't popular enough to have full production frameworks for distributed setups, so below are some other options for you to evaluate:

<img src="/resources/horovod.png" alt="Horovod" width="200"/>

> [Horovod](https://github.com/horovod/horovod) - built on top of MPI, but is designed specifically for TensorFlow, Keras, PyTorch, and Apache MXNet

<img src="/resources/deepspeed.png" alt="deepspeed" width="200"/>

> [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) - built on top of PyTorch, requires CUDA or ROCm, and is designed to optimize training for large models and thousands of GPUs in a distributed setup. It's a great choice if you're working with very large models

<img src="/resources/sagemaker.png" alt="sagemaker" width="200"/>

> [AWS Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html) - a managed service that makes it easy to train models on AWS infrastructure. It's a great choice if you're looking for a fully managed solution that takes care of the infrastructure for you

<img src="/resources/vertex.png" alt="vertex" width="200"/>

> [Google Vertex AI](https://cloud.google.com/vertex-ai/docs/training/distributed-training) - same as AWS but with Google infrastructure

## Final Conclusion

I hope you've enjoyed this walkthrough on distributed machine learning. We've covered the fundamentals that govern the principles around almost all distributed frameworks:

- MLX models and training
- Data curation
- Gradient averaging
- MPI

I hope that you've learned a lot and can make more informed decisions for your next machine learning project.

You'll find all of the scripts and files used inside this repo.
