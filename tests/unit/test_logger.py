import csv

from imdb_recommender.logger import ActionLogger


def test_logger(tmp_path):
    logger = ActionLogger(data_dir=str(tmp_path), batch_id="b")
    logger.log_rate("tt0480249", 9)
    logger.log_rate("tt0480249", 9)
    logger.log_watchlist("tt0111161", add=True)
    with open(tmp_path / "imdb_actions_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
