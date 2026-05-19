# Blind Image Deblurring with Deep Tensor Low-Rank Priors

This repository contains the code release for **Robust Non-Uniform Blind Deblurring with Deep Tensor Channel Correlation Prior**.

The method targets non-uniform blind deblurring, where both the latent sharp image and spatially varying blur kernels are estimated from a single blurred input image.

## Method Overview

We model non-uniform blur with a space-variant overlapped patch formulation. A deep image prior network generates the latent sharp image, while a pre-trained generative kernel network produces a grid of local blur kernels. The optimization is self-supervised: generated image and kernels are passed through the non-uniform blur operator and matched to the observed blurred image.

The core idea is a **Deep Tensor Channel Correlation Prior (DTCCP)**. Neighboring local blur kernels are strongly correlated under the SVOLA blur model, and RGB image channels also contain strong channel correlation. We therefore treat generated kernels and images as tensors and regularize the Tucker core tensor with a non-convex MCP penalty. This encourages sparse dominant tensor components while preserving important structures.

The implementation also includes **Candidate Kernel Initialization Refinement (CKIR)**. A pre-trained ResNet18 encoder first predicts an initial kernel latent code. Multiple noisy candidates are generated around this latent code, decoded into kernels, and scored by the self-supervised blur consistency objective. The best candidate is used to initialize kernel optimization.

The optimization follows an ADMM-style alternating scheme:

1. Initialize the image network input and kernel latent code.
2. Refine the initial kernel latent code with CKIR.
3. Alternately update the image network and kernel latent variable.
4. Apply Tucker decomposition and MCP thresholding to the generated image and kernel tensors.
5. Save intermediate and final deblurred images and estimated kernels.

## Repository Layout

```text
.
|-- mytest_nonuni_2tucker_lr=cos_finetune_paper_consistent_gpu.py
|-- compute_metrics_non_uniform_datasets.py
|-- eff.py
|-- SSIM.py
|-- networks/
|-- utils/
|-- metrics/
|-- scripts/
|-- datasets/
|   `-- lai/
|       |-- nonuniform/
|       `-- ground_truth/
`-- models/
    `-- lai/
```

The repository tracks source code only. Datasets, generated results, logs, and model weights are ignored by Git.

## Environment

The code was tested with Python and CUDA PyTorch. A typical environment includes:

```bash
pip install -r requirements.txt
```

The main dependencies are PyTorch, torchvision, tensorly, scipy, scikit-image, OpenCV, matplotlib, Pillow, tqdm, and joblib.

## Data and Weights

Place the Lai non-uniform benchmark images under:

```text
datasets/lai/nonuniform/
datasets/lai/ground_truth/
```

Place the pre-trained kernel decoder and encoder weights under:

```text
models/lai/netG_nonuniform.pth
models/lai/netE_nonuniform.pth
```

These files are not committed because they are data/model artifacts rather than source code.

## Single-GPU Inference

Run the paper-consistent version on the Lai non-uniform dataset:

```bash
python mytest_nonuni_2tucker_lr=cos_finetune_paper_consistent_gpu.py \
  --gpu 0 \
  --data_path ./datasets/lai/nonuniform \
  --save_path ./results_2tucker_newname \
  --num_iter 5000 \
  --save_frequency 500
```

The script saves intermediate outputs inside per-image work folders and writes the latest deblurred image to:

```text
results_2tucker_newname/<image_name>_x.png
```

This top-level output naming is compatible with `compute_metrics_non_uniform_datasets.py`.

## Multi-GPU Inference

For independent per-image parallelism, split the image set across multiple GPUs:

```bash
bash scripts/run_lai_parallel.sh "0 1 2 3" 5000 500
```

Arguments:

```text
scripts/run_lai_parallel.sh "<gpu ids>" <num_iter> <save_frequency>
```

Example:

```bash
bash scripts/run_lai_parallel.sh "3 4 5 7" 5000 500
```

The script creates a run folder under `parallel_runs/`, starts one process per GPU, and writes all deblurred images into `results_2tucker_newname/`.

## Metrics

After all expected `*_x.png` files are available, compute PSNR/SSIM on the Lai dataset:

```bash
python compute_metrics_non_uniform_datasets.py --dataset_name Lai
```

The script reads:

```text
datasets/lai/ground_truth/
results_2tucker_newname/
```

and writes aligned comparison outputs under `comparison/`.

## Notes

- The paper-consistent implementation uses MCP thresholding for the Tucker core tensor.
- CKIR uses 40 noisy candidates with noise standard deviation 0.3.
- Image and kernel optimization both use cosine annealing.
- The ADMM dual variables are kept across iterations within each image optimization run.
