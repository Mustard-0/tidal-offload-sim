"""True and reported locations use the same physical model at different inputs."""
from functools import lru_cache
import numpy as np

EARTH_RADIUS = 6378135.0
EARTH_MU = 3.986008e14
EARTH_RATE = 7.2921151467e-5
LIGHT_SPEED = 299792458.0


def satellites(cfg, time_s):
    return _satellites_cached(cfg["sat_height_m"], cfg["sat_inclination_deg"],
                              tuple(cfg["sat_overhead_s"]), time_s)


@lru_cache(maxsize=8192)
def _satellites_cached(height_m, inclination_deg, overhead_s, time_s):
    a = EARTH_RADIUS + height_m
    omega = np.sqrt(EARTH_MU / a ** 3)
    inc = np.deg2rad(inclination_deg)
    passes = np.asarray(overhead_s)
    ra = EARTH_RATE * passes
    at_pass = np.column_stack((np.cos(ra), np.sin(ra), np.zeros_like(ra)))
    direction = np.column_stack((-np.sin(ra) * np.cos(inc), np.cos(ra) * np.cos(inc),
                                 np.full_like(ra, np.sin(inc))))
    phi = omega * (time_s - passes)
    eci = a * (at_pass * np.cos(phi)[:, None] + direction * np.sin(phi)[:, None])
    angle = EARTH_RATE * time_s
    x = np.cos(angle) * eci[:, 0] + np.sin(angle) * eci[:, 1]
    y = -np.sin(angle) * eci[:, 0] + np.cos(angle) * eci[:, 1]
    return np.column_stack((y + 5000, eci[:, 2] + 5000, x - EARTH_RADIUS))


def candidate_nodes(cfg, region, time_s):
    center = np.asarray(cfg["region_centers_m"][region])
    bs = np.asarray(cfg["bs_xy_m"])
    radius = cfg["region_radius_m"]
    ground = np.linalg.norm(bs - center, axis=1) + radius <= cfg["bs_coverage_m"]
    # Conservative coverage margin for the full disk and the full lifetime.
    visible = np.ones(len(cfg["sat_overhead_s"]), dtype=bool)
    for t in np.linspace(time_s, time_s + cfg["lifetime_s"], 11):
        xyz = satellites(cfg, t)
        horizontal = np.linalg.norm(xyz[:, :2] - center, axis=1) + radius
        elevation = np.arctan2(xyz[:, 2], horizontal)
        speed_bound = 8500.0
        angular_margin = speed_bound / cfg["sat_height_m"] * cfg["lifetime_s"] / 20
        visible &= elevation - angular_margin >= np.deg2rad(cfg["sat_min_elevation_deg"])
    return np.r_[ground, visible]


def rate(cfg, xy, node, time_s, node_shares=1, user_shares=1):
    n_bs = len(cfg["bs_xy_m"])
    if node < n_bs:
        bx, by = cfg["bs_xy_m"][node]
        return _ground_rate_cached(float(xy[0]), float(xy[1]), bx, by,
                                   node_shares, user_shares,
                                   cfg["ground_gain_1m"], cfg["ground_path_exponent"],
                                   cfg["bandwidth_bs_hz"], cfg["user_power_w"],
                                   cfg["noise_psd_w_hz"])
    else:
        xyz = satellites(cfg, time_s)[node - n_bs]
        distance = max(float(np.linalg.norm(xyz - np.r_[xy, 0.])), 1.0)
        gain = (LIGHT_SPEED / (4 * np.pi * cfg["sat_frequency_hz"] * distance)) ** 2
        gain *= 10 ** ((cfg["sat_receive_gain_db"] - cfg["sat_extra_loss_db"]) / 10)
        bandwidth = cfg["bandwidth_sat_hz"] / max(node_shares, 1)
    power = cfg["user_power_w"] / max(user_shares, 1)
    snr = power * gain / (cfg["noise_psd_w_hz"] * bandwidth)
    return max(float(bandwidth * np.log2(1 + snr)), 1e-9)


@lru_cache(maxsize=65536)
def _ground_rate_cached(x, y, bx, by, node_shares, user_shares,
                        ground_gain, path_exponent, bandwidth_bs, user_power, noise_psd):
    distance = max(float(np.linalg.norm(np.asarray((x, y)) - (bx, by))), 1.0)
    gain = ground_gain * distance ** (-path_exponent)
    bandwidth = bandwidth_bs / max(node_shares, 1)
    power = user_power / max(user_shares, 1)
    snr = power * gain / (noise_psd * bandwidth)
    return max(float(bandwidth * np.log2(1 + snr)), 1e-9)


def propagation(cfg, xy, node, time_s):
    n_bs = len(cfg["bs_xy_m"])
    if node < n_bs:
        return 0.0
    xyz = satellites(cfg, time_s)[node - n_bs]
    return float(np.linalg.norm(xyz - np.r_[xy, 0.])) / LIGHT_SPEED
