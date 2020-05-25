import csv

from embedding_evaluation.process_benchmarks import process_benchmarks
from tqdm import tqdm

from riemann.embedding.conceptnet import standardized_uri

original_assertions_reduced = "/home/justin/research/conceptnet5/data/assoc/reduced.csv"
english_assertions = "/home/justin/research/riemannian-nlp/data/en_conceptnet_regularized_filtered.csv"

filter_wordsim = True

uris = []
if filter_wordsim:
    benchmarks = process_benchmarks()
    for _, benchmark in benchmarks.items():
        for (word1, word2), gold_score in tqdm(benchmark.items()):
            uris.append(standardized_uri("en", word1))
            uris.append(standardized_uri("en", word2))
uris = set(uris)

out_rows = []
with open(original_assertions_reduced, "r") as f:
    reader = csv.reader(f, delimiter=("\t"))
    for row in tqdm(reader):
        if row[0].startswith("/c/en/") and row[1].startswith("/c/en/"):
            if row[0] not in uris and row[1] not in uris:
                out_rows.append(row)

with open(english_assertions, "w+") as f:
    f.write("id1\tid2\tweight\n")
    for row in tqdm(out_rows):
        tab_s = "\t"
        f.write(f"{tab_s.join(row)}\n")
