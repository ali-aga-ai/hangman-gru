# Hangman Problem Report

A Brief Report for the Hangman Problem

Hi, I'm Ali. Over the past week, I've attempted to chip away at the hangman problem. It was one of the most engaging and challenging things I've worked on. This report describes the experiments and thought process behind my approach. I've also attached the code which gave me the best result. So let's begin!

## TL;DR

I tried the following methods:
- Uni-directional GRU (BASE) with Cross Entropy loss
- Bi-directional GRU
- Bi-directional GRU with positional embeddings
- BASE + Fine-tune on shorter words (since that's where it was failing)
- BASE + Weighted Loss Function (weighing infrequent characters more)
- BASE + New target probabilities (using n-grams)
- Oversampling rare-letter words

## Core Idea

The core idea was to predict a character based on the current masked word (e.g., `_A_` → C for the word CAT). This is conceptually close to sequence-to-sequence models like LLM next-token prediction. RNNs felt too simple, LSTMs too deep, and attention seemed overkill. So, GRU it is!

**The architecture.** We take in a masked word and try to predict the next character. Wrongly predicted letters should influence the next guess, so we add them to the GRU input [1]. The position of characters should also matter, so we experimented with positional embeddings [2].

**Target probabilities.** The model predicts the most likely next character. Consider a masked word where the true word is CAT, and the current state is `_A_`. We can estimate probabilities from the dictionary: for example, if we have CAR, CAT, RAT, MAT, then

```
P(C | _A_) = num occurrences of C in masked positions / total occurrences of all characters in masked positions = 2/8 = 0.25
```

For each masked word, we compute target probabilities as shown above [3].

**The dataset.** The dataset contains pairs of (masked word, target probability distribution). Each masked word goes through the GRU and outputs a softmax over all characters. At inference, we input the masked word, obtain softmax scores, and select the character with maximum probability [4].

This core architecture was used across all variations. The biggest bottleneck was compute: building the dataset and training it efficiently across GPUs. With over 200k words and 40 masked states per word, training on CPU was infeasible. I obtained access to three NVIDIA RTX A6000s, which required distributed data loading and parallel training [5].

## The Frequency Problem

However, the model struggled on unseen data, especially short sequences with infrequent letters [6]. The dataset was heavily skewed—e.g., 62 times more E than W [7]. Hence, the model preferred frequent letters, minimizing loss but failing full-word prediction.

I attempted to fine-tune my base model using only shorter words (length ≤ 7), but the frequency imbalance persisted. Cross-entropy loss favors frequent classes, since it minimizes expected log-likelihood by rewarding correct predictions on common letters more than rare ones [8].

**Evaluation metric.** The metric I used: for a target letter c, how often does the model correctly predict c out of all instances where c is the target? This can be seen as target accuracy [9]. Empirically, the model predicted frequent characters well (like E) but failed on rare ones (like W).

## Attempted Fixes

To improve per-letter accuracy, I tried oversampling words with rare letters and using a weighted loss [10]. While this slightly improved per-letter predictions, it did not translate to better overall game success.

During inference, I noticed some letters were predicted frequently but were about twice as likely to be incorrect [11]. This pattern persisted even with more training. To mitigate this, I applied a precision-based scaling: lowering probabilities for letters often predicted incorrectly (false positives) and boosting those that were more reliably correct [12].

Additionally, when the model wasn't confident—meaning the highest predicted probability was below 0.25—I treated it as uncertain. To guide the predictions, I blended in a simple frequency-based prior (freq-boost) favoring generally common letters. Already guessed letters, whether correct or wrong, were zeroed out in this prior to avoid repeated guesses. The blend-weight determined how much the model trusted the prior versus its own predictions, giving more weight to the prior when uncertainty was high.

I also experimented with ensembling multiple models, but this approach did not improve performance.

To tackle the disjoint nature of the test set, I tried a different method for constructing the training set. Instead of simple masking, I built target probabilities using corpus-derived positional and bigram frequencies. Revealed and wrong letters were excluded, small noise was added to prevent overfitting, and the distributions were normalized. This created generalizable probability distributions over letters, rather than being tied to specific words. Unfortunately, this approach did not yield better results.

I also experimented with adding length embeddings as an additional input to the GRU. The idea was to give the model explicit information about word length, hoping it would improve modeling of positional dependencies. In practice, this did not improve performance—per-letter accuracy and overall word prediction remained essentially the same. The GRU likely already captured sequence length implicitly, so the extra embedding provided little benefit.

Early on, I noticed the model struggled with short words. I fine-tuned the base model on words of length ≤ 7, but this also produced poor results.

I further experimented with a bi-directional GRU, hoping that processing the masked word forwards and backwards would allow context from both ends of the word to inform predictions—especially for rare letters. In practice, this modification did not meaningfully improve per-letter accuracy or full-word predictions. It slightly increased computation and memory usage, indicating that the unidirectional GRU was already sufficient to capture the necessary dependencies.

## Final Results

Overall, the best performing model in testing was the BASE model, checkpointed at 350 epochs (training continued up to 1500 epochs), which achieved a consistent win rate of approximately **55%** for 1000 test epochs.

## Conclusion

Despite extensive experimentation with architectural variations, loss reweighting, and inference adjustments, the best performance was a 55% win rate. This is under the expected 60% mark. Furthermore it performed even worse on the final recorded dataset.

The fundamental limitation across all approaches was the mismatch between cross-entropy loss—which optimizes average character prediction accuracy—and the true objective of complete word reconstruction. The model learned to favor frequent letters (E, A, S) because doing so minimized training loss, but this strategy failed on words requiring rare letters. If i had more time I would explore reinforcement learning or custom objectives that directly reward full-word success rather than per-character accuracy. I hope this report demonstrates both my technical approach and my ability to iterate through challenges—I'd welcome the opportunity to discuss these experiments and potential improvements in an interview.

---

## Details

### [1] Using the Wrong Letters

Let the masked input sequence be **x** = (x₁, x₂, ..., xₜ), and the corresponding wrong-letter vector **w** ∈ ℝ²⁶. The GRU encodes x as:

```
hₜ = GRU(x)
```

and we concatenate the hidden state with the wrong vector:

```
z = [hₜ; w]
```

Then,

```
y = Softmax(W₂ σ(W₁z + b₁) + b₂)
```

where **y** ∈ ℝ²⁶ is the probability distribution over characters.

### [2] Positional (Length) Embeddings

Let l be the length of the masked word. We embed it as:

```
eₗ = Embed(l)
```

and extend the previous concatenation:

```
z = [hₜ; w; eₗ]
```

then output probabilities via:

```
y = Softmax(W₂ σ(W₁z + b₁) + b₂)
```

Empirically, this didn't help—likely because the GRU already encodes position implicitly.

### [3] Target Probabilities

For a masked state M with revealed set R and wrong set W, define the valid candidate set:

```
C(M, W) = {v ∈ D : v matches mask M and has no letters in W}
```

Then, the target probability for letter c is:

```
P(c | M, W, R) = Σ_{v∈C(M,W)} 1{c ∈ v \ (R ∪ W)} / Σ_{v∈C(M,W)} Σ_{d∈v\(R∪W)} 1
```

### [4] Architecture

The full model:

```
x → Embedding → E(x) → GRU → hₜ → Concat → [hₜ, w, eₗ] → MLP → y = Softmax(W₂σ(W₁z))
```

### [5] Compute

Training required (1) building the masked dataset and (2) parallelizing across 3 GPUs to minimize wall-clock time.

#### 5.1 Building the masked training set

When I tried to generate millions of training states (as described in [3]) from a 227k-word vocabulary, the naive approach blew up: checking every candidate for every masked pattern was just too slow and memory-heavy, especially because each word could produce many sampled states (I used 40 samples per word).

To fix this, I attacked the problem on three fronts. First, I grouped words by length so candidate matching only ever looks at same-length words—that immediately cuts the search space. Second, I replaced set/list checks with compact bitmask integers for revealed and wrong letters (set to mask), so membership tests become single fast bitwise ops instead of loops. Third, I stopped treating each word as an independent task and instead split the vocabulary into 32 batches and ran them in parallel across CPU cores with `joblib.Parallel`, which balanced the load and avoided per-word process overhead.

I also added a simple cache for memoization: once a masked/wrong combination's target distribution is computed, it's memoized so repeated states don't recompute the same expensive candidate aggregation. The combined effect was dramatic: much smaller candidate lists, constant-time bit checks, parallel batch processing across many cores, and memoization of repeated work made the whole state-generation pipeline both fast and memory-efficient instead of painfully slow.

#### 5.2 The training run

When I started training the Hangman model on the full dataset, the setup looked good on paper: three NVIDIA RTX A6000 GPUs and millions of states. In practice, I quickly ran into a hard bottleneck—no multi-GPU speedup. Epochs were taking ~3 minutes each (and later even longer), and the GPUs were mostly idle waiting for data. I confirmed this by simple profiling (timers around data load vs compute) rather than fancy tools: data loading time dominated, and `nvidia-smi` showed under-utilization.

My early attempts were ad hoc. I tried tinkering with the learning rate manually and fiddled with batch sizes, but that only gave mixed results. I also experimented with gradient clipping and different optimizers; clipping had no notable effect. Searching for better LR schedules led me to `CosineAnnealingWarmRestarts` (found online); switching to it produced much more stable convergence than my manual hacks.

The real wins came from fixing the data pipeline. I pre-encoded the entire dataset (6.4M train states from the 160k word split) into tensors so the CPU didn't have to re-compute masks and target distributions every epoch (highest ROI). I switched the DataLoader to use persistent workers, non-blocking `.to(device)` transfers, and a reasonable prefetch factor so loading overlapped with GPU work (I mostly found these heuristics online). I also increased batch sizes (tried 2048 up to 16k) to improve GPU utilization, and used `nn.DataParallel` across the three A6000s. On the model side, I enabled mixed precision with `autocast` + `GradScaler`, which cut memory use and sped up the forward/backward passes by reducing fp precision.

After those changes, the pipeline behavior changed qualitatively: multi-GPU actually helped, per-epoch time fell to under 1 minute for the most optimized runs, and plateaus were typically reached around epoch 200. Checkpoints were saved every 50 epochs so I could resume if something failed. One practical problem I never fully solved was the Jupyter kernel instability on the VM I was using: it kept crashing and the VM didn't have the newer Python I wanted to switch to. Because it was the last day, instead of reconfiguring the VM and retraining from scratch, I focused on different experiments and analysis I had to do; resumptions were possible thanks to the checkpoints, but a clean, stable run on the latest Python would have been ideal.

In short: I diagnosed the bottleneck with simple timing, fixed the pipeline (pre-encoding, persistent workers, non-blocking transfers, prefetching, larger batches), used mixed precision and cosine restarts for training robustness, and cached/compressed per-sample work via bitmasks. Everything together turned a data-bound, slow multi-GPU run into a fast pipeline.

### [6] Failure on Short Words

Given input `___`, the model tends to guess frequent letters first (E, A), yielding wrong guesses repeatedly until some correct feedback appears—too late in the game. This limits win rate despite seemingly low loss.

### [7] Character Frequency Distribution

![Character frequency distribution showing E appears 62x more than W]

### [8] Cross-Entropy Bias

The dataset exhibits a strong character frequency imbalance. For example, the letter 'E' appears 62× more often than 'W'. When training with cross-entropy loss,

```
L_CE = -Σ_c p*(c) log p_θ(c)
```

the model is implicitly rewarded for assigning high probability to frequent characters. Since predicting 'E' lowers the loss more effectively than learning to predict rare characters like 'W', the model quickly learns that "E is safe." This drives the loss down, but it does not improve actual task performance.

This creates a critical mismatch between the training objective and the true task. Hangman requires full-word accuracy, which depends on correctly predicting every character in the word. For example, if the target word is "WAX" and the model predicts "EAX," the cross-entropy loss is still low because 'E' is statistically common, but the word prediction is incorrect, resulting in a game loss.

In short, cross-entropy optimizes average character accuracy, not full-word accuracy. As a result, the model overpredicts frequent letters, ignores rare ones, and achieves deceptively low loss without improving game performance. A reinforcement learning–based objective could address this by rewarding correct full-word predictions, but due to time constraints, this was not implemented, and the cross-entropy formulation was retained with this limitation noted.

### [9] Target Accuracy Metric

For each letter c,

```
Acc(c) = # times model predicts c correctly / # times c is the target letter
```

### [10] Weighted / Oversampled Loss

To counter dataset imbalance, I experimented with oversampling words containing rare letters or using a weighted loss. For instance, the following lines implement per-word weighting:

```python
# target_classes = target_probs.argmax(dim=1)
# weights = class_weights[target_classes]
# return -(target_probs * log_probs).sum(dim=1).mul(weights).mean()
```

Using this slightly improved per-letter accuracy, but did not yield significant gains in full-word prediction success.

### [11] Observing False-Positive Tendencies

During inference, I noticed that some letters were predicted frequently but were twice as likely to be incorrect (false positives). This trend persisted even as training progressed, as seen in the figure below.

### [12] Precision-Based Inference Adjustment

To mitigate persistent false positives, I computed per-character precision from test runs:

```python
correct = Counter({'e':117,'a':100,'i':81,...})
wrong = Counter({'s':102,'r':91,'t':89,...})
precision[ch] = correct[ch] / (correct[ch] + wrong[ch] + 1e-6)
```

I then multiplied the model's output probabilities by these precision weights during inference, in hopes that reweighting would favor letters that were historically predicted more reliably and suppress those frequently mispredicted. Unfortunately, this didn't work.