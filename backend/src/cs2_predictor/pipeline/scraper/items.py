import scrapy
from scrapy import Field, Item


class TeamItem(Item):
    hltv_id = Field()
    name = Field()
    country = Field()
    rank = Field()
    players = Field()


class MatchItem(Item):
    hltv_id = Field()
    team_a_id = Field()
    team_b_id = Field()
    team_a_name = Field()
    team_b_name = Field()
    format = Field()
    is_lan = Field()
    event_name = Field()
    scheduled_at = Field()
    map_pool = Field()


class ResultItem(Item):
    hltv_id = Field()
    team_a_id = Field()
    team_b_id = Field()
    team_a_name = Field()
    team_b_name = Field()
    team_a_score = Field()
    team_b_score = Field()
    format = Field()
    event_name = Field()
    played_at = Field()
    maps = Field()


class MatchDetailItem(Item):
    hltv_id = Field()
    team_stats = Field()
    map_results = Field()
