# kolos - A supervised machine learning tool for collocation extraction

This tool orders collocation candidates, extracted from a corpus via morphosyntactic patterns, by using supervised machine learning model trained on manually annotated collocation candidates.

It consists of two scripts
- `train.py`, which trains the necessary models in the `models/` directory, based on KOLOS data.
- `run.py`, which runs the models over the KAS data.

The `run.py` script has two modes - `--eval` for per-gramrel evaluation (of KOLOS models over KAS data), and `--output` for the per-headword ranking of the KAS candidates given the KOLOS model.

The `run.py` script was already run in the following manner: `python run.py --eval > eval.txt` and `python run.py --output > output.txt`, meaning that the output of the script can already be explored in the `eval.txt` and `output.txt` files.
