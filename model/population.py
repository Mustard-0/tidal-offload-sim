"""Actual public region membership, including unequal active populations."""


def region_counts(cfg):
    counts = cfg.get('region_user_counts', [cfg['users_per_region']]*len(cfg['region_centers_m']))
    if len(counts) != len(cfg['region_centers_m']) or any(int(n)!=n or n<1 for n in counts):
        raise ValueError('Expected positive integer active-user counts per region')
    return tuple(int(n) for n in counts)


def user_region(cfg, user):
    if int(user)!=user or user<0:
        raise ValueError('Invalid user ID')
    offset = 0
    for region, count in enumerate(region_counts(cfg)):
        if offset <= user < offset+count:
            return region
        offset += count
    raise ValueError('User outside actual active population')
