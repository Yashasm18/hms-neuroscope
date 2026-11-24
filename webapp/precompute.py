"""
Precompute seizure-score caches for the bundled CHB-MIT recordings.

    python -m webapp.precompute

Writes results/score_cache/<recording>-<model hash>.npz. The web app then answers
detection requests for these recordings instantly, which matters on small hosts.
"""

from webapp import service as S

if __name__ == "__main__":
    for path in S.precompute_bundled_scores():
        print("wrote", path)
