# ATML PA3 — Fine-Tuning, Vision-Language Models, and Test-Time Compute

EE-5102 / CS-6304, Fall 2026.

## Repository layout

```
configs/            run configurations (YAML), one per training run
src/common/         seeding, paths, logging, resumable checkpoints
src/task1/          LLM fine-tuning: Full FT, L2-SP, LoRA-4/16, OPLoRA-16
src/task2/          SigLIP–Qwen VLM and grounding evaluation
src/task3/          test-time compute: selection, refinement, adaptive sampling
scripts/            entry points run from Kaggle notebooks
splits/             saved example, pair and split IDs
results/            machine-readable results (JSON/CSV) behind every reported number
```