from datasets import load_dataset

my_cache_dir = "C:\programlearning\Re-DeepReasearcher\data"

ds = load_dataset("AI-MO/NuminaMath-CoT",cache_dir=my_cache_dir)

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("open-r1/codeforces-cots", "solutions_decontaminated", cache_dir=my_cache_dir)

ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki",cache_dir=my_cache_dir)

ds = load_dataset("allenai/openbookqa", "additional",cache_dir=my_cache_dir)
