"""Compatibility-only correction for the inherited transport plan field."""
from scalability_protocol import *
from scalability_protocol import plan as original_plan


CAMPAIGN = 'tt-mappo-matched-scale-up-20261119-gatefix'
BRANCH = 'experiment/' + CAMPAIGN
SOURCE_BASE = 'd87203e3b49d3e1590313b66f12e6711efb037b5'


def plan():
    value = original_plan()
    value['inherited_review_hashes'] = value['inherited_code']
    value['source_base'] = SOURCE_BASE
    value['schema'] = 'matched-scale-up-v2-transport-compatibility'
    return value
