from cs2_predictor.pipeline.scraper.items import TeamItem, MatchItem, ResultItem, MatchDetailItem


def test_items_import():
    assert TeamItem is not None
    assert MatchItem is not None
    assert ResultItem is not None
    assert MatchDetailItem is not None
