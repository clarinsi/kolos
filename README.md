# kolos - A supervised machine learning tool for collocation extraction

This tool orders collocation candidates of a specific headword, for a specific gramrel (morphosyntactic pattern), extracted from a corpus, by using supervised machine learning model trained on manually annotated collocation candidates. The tool uses static lemma embeddings (for Slovenian the CLARIN.SI-embed.sl embeddings from http://hdl.handle.net/11356/1204) as explanatory variables, aiming at learning the semantic specificities of good and bad collocates.

The tool consists of two scripts
- `train.py`, which trains the necessary models in the `models/` directory, based on KOLOS data (available in `kolos.csv`).
- `run.py`, which runs the models over the KAS data (available in `kas.csv`).

The `run.py` script has two modes - `--eval` for per-gramrel evaluation (of KOLOS models over KAS data), and `--output` for the per-headword ranking of the KAS candidates given the KOLOS model.

The `run.py` script was already run in the following manner: `python run.py --eval > eval.txt` and `python run.py --output > output.txt`, meaning that the output of the script can already be explored in the `eval.txt` and `output.txt` files.

The evaluation is performed with ROC AUC, meaning that a 0.5 result shows a random ordering, while a 1.0 result shows a perfect ordering.
 
