import numpy as np

# Path to your TSV file
tsv_path = "gtex_gene_expected_count"
sample_nrows = 10  # Change this to however many rows you want to sample

# Step 1: Count number of columns
with open(tsv_path, "r") as f:
    header = f.readline()
    num_columns = len(header.rstrip("\n").split("\t"))
print(f"Number of columns: {num_columns}")