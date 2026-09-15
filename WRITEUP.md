# Task 1 — Sparse Attention from Scratch

## 1. What I built

The implementation has three attention paths. The first is a manual dense reference. The other two use exactly the same scaled dot-product equations, but restrict the set of keys before computing the scores.

The dense version is intentionally uncomplicated. Given `q`, `k`, and `v`, it computes `QK^T`, divides by `sqrt(d)`, applies the boolean mask, takes the softmax, and multiplies by `V`. I did not use PyTorch's fused scaled-dot-product attention because the task specifically asks for a hand-written reference.

For the first sparse pattern I used causal sliding-window attention. With window size `w`, query position `i` can see positions `i-w+1 ... i`. This changes the score tensor from roughly `N x N` to `N x w`.

For the second pattern I used a block layout with three ingredients: local neighboring blocks, a small global token set, and deterministic random blocks. I kept the random selection seeded so that the mask is reproducible. The causal constraint is applied after choosing the block connections, so no query can see a future token.

This is close in spirit to the local/global/random construction used by BigBird. The paper's motivation is that sparse structure can keep the useful communication paths of full attention while removing most of the quadratic work. One of the paper's observations is that a small number of global tokens can be unusually important for maintaining long-range connectivity.

## 2. Correctness and the subtle point about the oracle

The sparse implementations are not supposed to match unrestricted dense attention. That would be the wrong test: a sliding window deliberately throws information away.

Instead, correctness is tested like this:

```text
same Q, K, V
      |
      +---- manual dense attention + sparse mask -> reference
      |
      +---- sparse implementation             -> candidate
```

The maximum absolute difference for both sparse implementations was about `3.6e-7` in the included harness, which is comfortably within the `1e-5` tolerance used by the tests.

There is a second failure mode that is easy to miss. If a query has no legal keys, the masked row is all `-inf`. A normal softmax then becomes `0 / 0` and returns NaN. I handle that explicitly in `safe_softmax()`: completely masked rows get an all-zero probability vector, and therefore an all-zero output vector. The issue can occur naturally when sparse structure and causal masking interact at boundaries, so it is not just a synthetic test case.

## 3. Benchmark setup

The benchmark measures forward-pass wall time and peak process memory. I used the same random tensors and the same attention dimensions for every pattern in a run. For the CPU results committed in this repository I used one head with dimension 16, window size 64, block size 32, one local neighbor block, one random block, and eight global tokens. The small head dimension keeps the 8192 dense case practical on the available CPU.

The measured CPU results were:

| N | Dense time (s) | Sliding time (s) | Block time (s) |
|---:|---:|---:|---:|
| 512 | 0.0030 | 0.0053 | 0.1115 |
| 1024 | 0.0078 | 0.0049 | 0.0383 |
| 2048 | 0.0245 | 0.0075 | 0.0633 |
| 4096 | 0.1826 | 0.0144 | 0.1495 |
| 8192 | 1.0892 | 0.0255 | 0.2277 |

The crossover is the interesting part. At small `N`, sparse attention can be slower because the implementation has some indexing and gather overhead. Once the quadratic dense score matrix gets large enough, the difference becomes much more visible. At `N=8192`, sliding-window attention was roughly 43x faster than the dense reference in this configuration, while block-sparse attention was about 4.8x faster.

The memory plot shows the same qualitative trend. The absolute CPU process numbers include the Python/PyTorch baseline, so the ratios should be treated as relative evidence rather than as a clean accounting of only the attention tensor. The important point is that the sparse score tensor scales with the number of allowed keys rather than `N^2`.

I would use the same script with `--device cuda --dtype float16` on the T4 for the final hardware-specific submission numbers.

## 4. What each pattern loses

Sliding-window attention is very good at local structure. Character-level language modeling needs a lot of local information: spelling, punctuation, nearby syntax, and short-range rhythm. A small window therefore preserves a surprising amount of basic signal.

The weakness is long-range communication. A token near the end of a long sequence cannot directly inspect an early token. Information has to move through intermediate positions and layers. With only two Transformer layers, this limitation is particularly noticeable because there are not many opportunities to propagate a distant fact.

The block-sparse pattern trades a little more complexity for better connectivity. Local blocks still carry the nearby information, while the global tokens provide a cheap route across the entire sequence. Random blocks add a few additional shortcuts that are not tied to distance. This is the part that matters disproportionately: even a small number of global positions can connect otherwise distant parts of the sequence.

The downside is that global tokens are a bottleneck. They do not recreate all the pairwise interactions of dense attention; they simply give the network a small number of places through which long-range information can pass. Random links help, but they are not guaranteed to connect the most useful pairs for a particular example.

## 5. TinyShakespeare experiment

The repo contains a small character-level GPT with two Transformer layers and three switchable attention implementations. The same model configuration, initialization seed, optimizer, batch stream, and validation procedure are used for each pattern. The only intended architectural difference is the attention pattern.

The training command is:

```bash
python gpt/train.py --device cuda --steps 2000
```

and the results are written to `results/gpt/quality_results.json` with a validation-loss plot generated by `gpt/evaluate_quality.py`.

I was able to verify the model forward pass for all three modes and verify the attention implementations with the correctness harness. I could not honestly include a full TinyShakespeare loss table from this environment because the execution environment has no external network access and does not contain the dataset locally, and it also does not provide a CUDA T4. The training code is therefore included and reproducible, but the final numeric quality comparison needs to be run in Colab where the dataset can be downloaded and the intended compute is available.

That limitation is preferable to filling the writeup with fabricated loss numbers.

## 6. Takeaways

The main result is not just that sparse attention is faster. It is that the speedup comes from making the allowed interaction pattern explicit and then **not constructing scores for disallowed pairs**. A masked dense matrix is still quadratic even if most of it is `-inf`.

Sliding windows give the cleanest compute savings, but they are also the easiest way to lose long-range information. The block-sparse pattern is a more interesting compromise because global and random links provide communication paths without returning to a full `N x N` matrix.

For a small causal language model I would expect dense attention to remain the quality reference, sliding windows to lose some long-range behavior, and local+global+random sparsity to recover some of that loss at a substantially lower attention cost. The exact amount of quality retained is what the TinyShakespeare run is intended to measure rather than assume.
