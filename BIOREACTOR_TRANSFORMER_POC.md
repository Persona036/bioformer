# Bioreactor Transformer POC Plan

Last updated: March 30, 2026

## Goal

Build a fast, credible, low-budget proof of concept for a transformer-based model that learns from bioreactor process data and predicts a scientifically meaningful outcome.

The right first target is not "build a foundation model." The right first target is:

`early process history -> final batch outcome`

For the first POC, that outcome should be a single scalar with clear operational value and a public dataset that is easy to access.

## Executive Recommendation

### Primary recommendation

Use the **Erythromycin fermentation process dataset** as the first supervised dataset.

Why:

- It is small and practical for an individual researcher.
- It has a clear target variable: **chemical potency during fermentation**.
- It includes **406 fermentation batches**, sampled **hourly**, with average length around **154 hours**.
- It includes **23 process variables** spanning conditions, cumulative feeds, physicochemical indicators, and biochemical indicators.
- The dataset is published on Zenodo under **CC BY 4.0**.

Recommended first task:

`first 24/48/72 hours of a batch -> final potency`

### What not to do in phase 1

- Do not combine heterogeneous datasets into one supervised table on day 1.
- Do not try to reproduce a TRIBE-scale training stack.
- Do not train a large multimodal transformer from scratch before proving signal exists.

### Expansion recommendation

After the first POC works, expand to **NISTCHO** for a stronger biopharma story:

- upstream feeding strategy
- culture performance
- product quality attributes

That gives a more scientifically interesting phase 2, but it is heavier to preprocess and much more complex than the erythromycin dataset.

## Dataset Strategy

### Phase 1 dataset: use this first

**Dataset:** Erythromycin fermentation process dataset  
**URL:** https://doi.org/10.5281/zenodo.14619074

What the public metadata says:

- Historical production data from 2022
- 406 independent fermentation batches
- Hourly sampling
- Average batch length about 154 hours
- 23 process variables
- Target variable is chemical potency during fermentation

This is the best first POC dataset because it is small enough to handle cheaply but large enough to test whether a sequence model can learn batch dynamics.

### Phase 2 dataset: add this later

**Dataset:** NISTCHO feeding strategies / cNISTmAb product quality data  
**URLs:**  
https://doi.org/10.5281/zenodo.17046014  
https://doi.org/10.5281/zenodo.18609042

Why it matters:

- It is much closer to real biopharma process development.
- It connects feeding strategy and culture performance to product quality attributes.
- It supports targets like titer, productivity, and glycosylation-related CQAs.

Why it is not phase 1:

- The data package is large.
- Preprocessing burden is much higher.
- Product-quality targets are more heterogeneous than potency.

### Optional phase 2.5 datasets

Use these only after phase 1, and mostly for self-supervised pretraining or anomaly detection:

- **Lab-scale membrane bioreactor high-frequency online data**  
  https://doi.org/10.5281/zenodo.821585
- **Dynamic compartment models for fed-batch fermentations**  
  https://doi.org/10.4121/0a08d2ec-8959-403f-afea-2b085dc9f3a6.v1

These are useful for process-dynamics modeling, but they are not the cleanest first supervised benchmark for batch potency prediction.

## Should Datasets Be Combined?

### Phase 1 answer

No. Train on the erythromycin dataset alone first.

Reason:

- same organism/process family
- same target type
- same logging regime
- low schema reconciliation cost

### Phase 2 answer

Yes, but only with the right objective.

Recommended multi-dataset strategy:

- Pretrain a transformer with **self-supervised objectives** across multiple datasets.
- Add a `dataset_id` embedding so the model knows which source it is reading.
- Fine-tune separate heads per dataset or per target family.

Do **not** naively merge EFP and NISTCHO into one supervised target table. They are too different in biology, operating regime, and endpoint meaning.

## What the Model Should Predict

### Primary supervised target

**Task A:** early-to-final regression

`process sequence up to hour H -> final batch potency`

Suggested horizons:

- hour 24
- hour 48
- hour 72

This gives you an "early warning" story that scientists and process engineers care about.

### Secondary targets

**Task B:** sequence-to-sequence forecasting

`history up to time t -> next 12/24 hours of potency or key variables`

**Task C:** masked reconstruction

`input sequence with masked values -> reconstruct missing values`

This is useful as a self-supervised pretraining objective.

**Task D:** batch embedding and retrieval

`whole batch sequence -> fixed-length latent vector`

Use this for:

- clustering good vs bad runs
- nearest-neighbor retrieval
- anomaly detection

## Recommended Input/Output Encoding Format

Treat the raw CSV as a source file, then convert it into a canonical training format.

### Canonical sample schema

Each batch should become one sample:

```text
sample = {
  batch_id: str,
  dataset_id: int,
  x_num: float32[T, F_num],
  x_mask: uint8[T, F_num],
  time_hours: float32[T],
  batch_meta_num: float32[M_num],
  batch_meta_cat: int64[M_cat],
  y_final: float32[1],
  y_seq: float32[T, Y]  # optional
}
```

### Tensor view after batching

```text
x_num        : [B, T, F_num]
x_mask       : [B, T, F_num]
time_hours   : [B, T]
meta_num     : [B, M_num]
meta_cat     : [B, M_cat]
y_final      : [B, 1]
y_seq        : [B, T, Y]  # optional
```

### Recommended EFP-specific encoding

For the erythromycin dataset:

- `T`: 168 or 192 timesteps max
- `F_num`: 23 numeric variables
- `M_num`: 0 to small number
- `M_cat`: 1 or 2 if you add run metadata or split markers

If a batch is shorter than `T`, pad it and keep a padding mask.

### Preprocessing rules

- Group strictly by `batch_id`.
- Sort within batch by time.
- Fit normalization using **training batches only**.
- Use variable-wise z-score normalization.
- Keep a missing-value mask instead of hiding missingness.
- For cumulative feed variables, keep both the raw cumulative values and their first differences if useful.
- Never split rows from the same batch across train and validation.

## Recommended Model

### Important clarification

TRIBE v2 is a transformer-based model, but it is not the right scale target for this project. TRIBE depends on expensive pretrained encoders and a large downstream head. Your POC should use a **small encoder-only time-series transformer**.

### POC model architecture

Recommended architecture:

```text
numeric features
  -> linear projection to d_model
  -> add time embedding
  -> add dataset embedding
  -> transformer encoder
  -> masked pooling or CLS token
  -> prediction head
```

### Suggested starting configuration

- Model type: encoder-only transformer
- Sequence length: 168 or 192
- Feature count: 23 numeric features
- `d_model`: 128
- `n_heads`: 4
- `n_layers`: 4
- Feed-forward width: 256 or 512
- Dropout: 0.1
- Pooling: mean pooling over valid timesteps, or a CLS token
- Head: 2-layer MLP to scalar output

This is a small model by transformer standards and should fit comfortably on a 24 GB GPU.

### Optional phase 2 configuration

If the first model works:

- `d_model`: 192 or 256
- `n_layers`: 6
- Add a multitask head
- Add masked-value reconstruction during pretraining

### What I would not start with

- `d_model >= 512`
- more than 8 layers
- giant context windows
- multimodal fusion
- end-to-end pretraining from unstructured raw text, images, or omics

## Recommended Training Stack

### Libraries

- `pandas` for raw wrangling
- `numpy` for array handling
- `pyarrow` or `parquet` for cached processed data
- `pytorch`
- `pytorch-lightning` or plain PyTorch
- `scikit-learn` for batch splits and baselines
- `xgboost` or `lightgbm` for a strong non-transformer baseline

### Strongly recommended baseline ladder

Before trusting the transformer, train these:

1. ElasticNet or ridge regression on summary features
2. LightGBM or XGBoost on summary features
3. Small transformer on raw sequences

If the transformer cannot beat LightGBM, the issue is usually:

- not enough data
- bad target choice
- leakage in preprocessing
- unstable train/validation split

## Training Procedure

### Step 1: define the first task

Use:

`first 48 hours -> final potency`

Then repeat for 24 and 72 hours.

### Step 2: split correctly

Use one of:

- 70/15/15 split by batch
- 5-fold cross-validation by batch

Do not do random row-level splitting.

### Step 3: prepare windows

For each batch:

- keep hours `0:H`
- pad to a fixed maximum length
- produce a mask tensor
- store `y_final`

### Step 4: train with conservative settings

Recommended starting hyperparameters:

- optimizer: AdamW
- learning rate: `3e-4`
- weight decay: `1e-2`
- batch size: `32` or `64`
- epochs: `50-100`
- early stopping patience: `10`
- mixed precision: yes
- gradient clipping: `1.0`

### Step 5: evaluate

Track:

- MAE
- RMSE
- R-squared
- Pearson correlation
- Spearman correlation

Also plot:

- predicted vs true final potency
- error by horizon
- error by batch duration
- embedding clusters

## Compute Requirements

## Minimum viable compute

For preprocessing and feature baselines:

- modern laptop CPU is enough
- 16 to 32 GB system RAM is comfortable
- no GPU required

## Comfortable transformer POC compute

Recommended target:

- **1 GPU with 24 GB VRAM**
- 32 GB system RAM on the host
- local SSD or attached block storage

This is enough for:

- sequence length under 256
- `d_model` 128 to 256
- 4 to 6 transformer layers
- batch sizes in the 16 to 64 range

## Likely runtime

For the EFP dataset with the recommended small transformer:

- a single clean training run should usually be in the range of **1 to 4 GPU-hours**
- a sensible tuning cycle of multiple horizons, seeds, and ablations is more like **20 to 60 GPU-hours total**

That is still an individual-budget project.

## Recommended places to train

### Option 1: local machine

Best if you already own an NVIDIA GPU with 12 to 24 GB VRAM.

Use local for:

- preprocessing
- baselines
- small smoke tests

### Option 2: Google Colab

Use for:

- notebook-based experiments
- smoke tests
- one-off debug runs

Do not rely on free Colab for the full POC.

Why:

- Google states that free Colab gives access to GPUs and TPUs at no cost, but resources are **not guaranteed**, **not unlimited**, and free notebooks run at most **12 hours** depending on availability and usage.
- Colab Pro+ can run up to **24 hours** if you have enough compute units.

Official Colab FAQ:

- https://research.google.com/colaboratory/faq.html?hl=en

### Option 3: Lambda Cloud

Good if you want straightforward instance-based pricing from an official public price page.

Lambda's public pricing page currently lists these instance prices, **effective April 6, 2026**:

- Quadro RTX 6000, 24 GB: **$0.69 per GPU-hour**
- A10, 24 GB: **$1.29 per GPU-hour**
- A6000, 48 GB: **$1.09 per GPU-hour**
- A100 PCIe, 40 GB: **$1.99 per GPU-hour**

Official pricing page:

- https://lambda.ai/pricing

### Option 4: Runpod

Good if you want flexible GPU marketplace options and are willing to compare availability manually at booking time.

Runpod's public pricing page currently shows 24 GB, 32 GB, and 48 GB GPU classes including:

- 24 GB: L4, RTX 3090, RTX 4090, RTX A5000
- 32 GB: RTX 5090
- 48 GB: A40, A6000, RTX 6000 Ada, L40, L40S

Official pricing page:

- https://www.runpod.io/pricing

For this project, a 24 GB or 48 GB class GPU is the practical sweet spot.

## Rough Budget

These are rough planning numbers, not guarantees.

### Lowest-cost path

- preprocess locally
- run baselines locally
- use free Colab only for smoke tests

Estimated cost:

- **$0 to $10**

This is enough to prove the dataset and code path work, but not enough for reliable iteration.

### Recommended comfortable POC

Use a paid cloud GPU for the transformer, preferably 24 GB or 48 GB VRAM.

Assume:

- 20 to 60 total GPU-hours
- small transformer
- several horizons
- a few seeds

Budget using current public Lambda rates:

- 24 GB Quadro RTX 6000 at $0.69/hr: **$14 to $41**
- 24 GB A10 at $1.29/hr: **$26 to $77**
- 48 GB A6000 at $1.09/hr: **$22 to $65**

Add storage and incidental compute:

- dataset storage and artifacts: **$5 to $15**

Recommended working budget:

- **$30 to $80 total**

### More serious iteration budget

If you start doing:

- 5-fold CV
- larger models
- multi-dataset pretraining
- heavier hyperparameter search

Budget:

- **$100 to $300**

That is still realistic for an individual if scoped carefully.

## What Scientists Might Care About

A useful first paper or internal demo does not need to claim a universal bioprocess foundation model. It can be valuable if it shows any of the following:

- early prediction of final potency from partial process history
- which process variables matter most at different phases
- whether latent batch embeddings separate good and bad runs
- whether the model identifies abnormal trajectory regimes early
- whether pretraining on mixed process datasets improves fine-tuning on a target process

The strongest "scientific" angle in phase 1 is:

`a transformer learns latent fermentation-state trajectories that improve early potency prediction and batch similarity analysis`

## Deliverables for the First POC

The first successful deliverable should include:

- batch-level preprocessing pipeline
- baseline model results
- transformer model results
- holdout plots
- learned batch embeddings
- a short comparison of 24h, 48h, 72h forecast horizons

Nice-to-have additions:

- attention visualizations
- integrated gradients or ablation plots
- nearest-neighbor retrieval of similar batches

## Risks and Guardrails

### Main risks

- too few independent batches for a large model
- leakage between train and validation
- target not stable or too noisy
- overfitting through repeated manual tuning
- confusing interpretability with causality

### Guardrails

- split by batch only
- keep strong baselines
- lock one clean validation protocol early
- report uncertainty across multiple random seeds
- treat importance analyses as associational, not causal

## Recommended Project Structure

```text
project/
  data/
    raw/
      efp/
        EFP_long.csv
    processed/
      efp/
        train.parquet
        val.parquet
        test.parquet
  notebooks/
    01_eda.ipynb
    02_baselines.ipynb
    03_transformer.ipynb
  src/
    datasets/
      efp.py
    models/
      transformer.py
      baselines.py
    train/
      train_baseline.py
      train_transformer.py
    eval/
      metrics.py
      plots.py
  configs/
    baseline.yaml
    transformer_small.yaml
```

## Recommended Milestones

### Week 1

- ingest EFP
- build batch-level splits
- produce summary features
- train ridge and LightGBM baselines

### Week 2

- build small transformer
- train on 48-hour prediction
- run 3 seeds
- compare against baselines

### Week 3

- add 24h and 72h horizons
- produce embedding and retrieval analyses
- write a short report

### Week 4

- decide whether to expand to NISTCHO
- if yes, build a separate fine-tuning pipeline or self-supervised pretraining corpus

## Final Recommendation

If the objective is a **fast, low-budget, scientifically credible transformer POC**, the right path is:

1. Use the **Erythromycin fermentation process dataset** alone.
2. Build a **small encoder-only transformer** with 4 layers and `d_model = 128`.
3. Train it on **early-to-final potency prediction**.
4. Run preprocessing and baselines locally.
5. Train the transformer on a **24 GB or 48 GB cloud GPU**.
6. Budget **$30 to $80** for a comfortable first iteration.
7. Expand to **NISTCHO** only after the first POC demonstrates real signal.

This gives you the best balance of cost, speed, and scientific usefulness.

## Sources

- Erythromycin fermentation process dataset: https://doi.org/10.5281/zenodo.14619074
- NISTCHO supporting data v1: https://doi.org/10.5281/zenodo.17046014
- NISTCHO supporting data newer record: https://doi.org/10.5281/zenodo.18609042
- Lab-scale membrane bioreactor high-frequency online data: https://doi.org/10.5281/zenodo.821585
- Dynamic compartment models for fed-batch fermentations: https://doi.org/10.4121/0a08d2ec-8959-403f-afea-2b085dc9f3a6.v1
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html?hl=en
- Lambda Cloud pricing: https://lambda.ai/pricing
- Runpod pricing: https://www.runpod.io/pricing
