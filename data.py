"""Ingestion: pybaseball data from Statcast and Baseball Reference to one player-half panel.

One row per (BatterID, season half), carrying the plate-appearance outcomes, the pitch-level
counts behind the PA stage's features, and the batted-ball counts behind the BIP
stage's features.

`build_people` is the one function that does not touch the panel. Names come off the
Chadwick register by MLBAM id, which is how a hitter with no panel row still gets a name.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from pybaseball import batting_stats_range, playerid_reverse_lookup


# Summation of ball-in-play level features
BIP_SUMS = ["nHardHit", "nAir", "nSwSp", "nPull", "nPullAir", "sumEV", "sumEV90"]

# wOBA weights provided by original assignment
WOBA_WEIGHTS = {"walk": 0.689, "hit_by_pitch": 0.72, "single": 0.882,
                "double": 1.254, "triple": 1.59, "home_run": 2.05}



CACHE_DIR = Path(__file__).parent / "cache"


def load_statcast(start_dt='2024-01-01', end_dt='2025-12-31', refresh=False): 
    """
    Pull Statcast data using pybaseball, caching the raw pull to a parquet file in order
    to avoid re-downloading the same data.
    
    Filters to regular season games only.

    Parameters:
        start_dt : start date string (YYYY-MM-DD)
        end_dt   : end date string (YYYY-MM-DD)
        refresh  : re-pull from Statcast even if a cached copy exists
    Returns:
        Raw statcast DataFrame
    """
    cache = CACHE_DIR / f"statcast_{start_dt}_{end_dt}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
    else:
        from pybaseball import statcast  # imported lazily; only needed for live pulls
        df = statcast(start_dt=start_dt, end_dt=end_dt)
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_parquet(cache)

    df = df[df['game_type'] == 'R'].copy()
    df["spray_angle"] = np.arctan((df['hc_x'] - 125.42) / (198.27 - df['hc_y'])) * (180 / np.pi)
    df["season_half"] = np.where(df["game_date"].dt.month < 7, 1, 2)
    df.rename(columns={"batter": "BatterID"}, inplace=True)
    return df


def season_id(season, half): 
    """Contiguous half index: 1 = 2024 1H ... 4 = 2025 2H, 5 = 2026 1H.

    Args:
        season (Series or int): calendar year, 2024 or later
        half (Series or int): 1 for the first half, 2 for the second
    Returns:
        sid (Series or int): the contiguous half index, 1 through 5
    """
    return (season - 2024) * 2 + half


def build_pa(): 
    """
    Calculate the plate-appearance counts and outcomes for each player-half, including wOBA.

    Args:
        None
    Returns:
        pa_counts (DataFrame): one row per (BatterID, Season, SeasonHalf), with PA counts and outcomes
    """

    # Baseball-Reference date ranges include the postseason, so each 2nd half ends on the
    # last regular-season day. 
    halves = [
        ("2024-03-01", "2024-06-30", 2024, 1),
        ("2024-07-01", "2024-09-30", 2024, 2),
        ("2025-03-01", "2025-06-30", 2025, 1),
        ("2025-07-01", "2025-09-28", 2025, 2),
    ]

    frames = []
    for start_dt, end_dt, season, season_half in halves:
        df = batting_stats_range(start_dt, end_dt)
        df = df[df["PA"] > 0]
        df["BatterID"] = df["mlbID"]
        df["game_year"] = season
        df["season_half"] = season_half

        df["BIP"] = df["AB"] - df["SO"] + df["SF"] + df["SH"]
        df["1B"]  = df["H"] - df["2B"] - df["3B"] - df["HR"]
        # For the purposes of this project, we will treat all non-base-hit balls in play as outs. 
        # This is a simplification, but it allows us to focus on the outcomes that are most 
        # relevant to wOBA and player performance.
        df["OUT"] = df["BIP"] - df["H"]
        df["K"]   = df["SO"]

        counts = df[["PA", "BIP", "1B", "2B", "3B", "HR", "OUT", "K", "HBP", "BB"]].add_prefix("n")

        # wOBA is a rate but we need to use the raw counts for modeling.
        w = WOBA_WEIGHTS
        counts["nwOBA"] = w["walk"] * counts["nBB"] + w["hit_by_pitch"] * counts["nHBP"] + \
            w["single"] * counts["n1B"] + w["double"] * counts["n2B"] + \
            w["triple"] * counts["n3B"] + w["home_run"] * counts["nHR"]
        counts["wOBA"] = counts["nwOBA"] / counts["nPA"]

        frames.append(df[["BatterID", "game_year", "season_half"]].join(counts))

    return pd.concat(frames, ignore_index=True)



def build_player_half(statcast_df, pa_counts): 
    """One row per (player, season half): outcomes, pitch counts, batted balls, SeasonID.

    Combines the PA counts from build_pa() with the batted-ball counts from the Statcast data to
    create the player-half panel.

    Args:
        statcast_df (DataFrame): the raw statcast DataFrame
        pa_counts (DataFrame): the PA counts DataFrame from build_pa()
    Returns:
        ph (DataFrame): one row per (BatterID, Season, SeasonHalf)
    """

    sc = statcast_df.copy()

    # First filter for balls in play. Then every batted-ball feature is one
    # indicator summed over those balls.
    #
    # Handedness: -45 degrees is the 3B line, so pull is negative for a RHB and positive
    # for a LHB
    bb = sc[sc["type"] == 'X'].copy()  # filter for balls in play
    bb["HardHit"] = bb["launch_speed"] >= 95
    bb["Air"] = bb["launch_angle"] >= 10
    bb["SwSp"] = bb["launch_angle"].between(8, 32)  # Sweet Spot
    bb["Pull"] = (((bb["stand"] == "L") & bb["spray_angle"].between(15, 45)) |
                  ((bb["stand"] == "R") & bb["spray_angle"].between(-45, -15)))
    bb["PullAir"] = bb["Pull"] & bb["Air"]
    bb = bb.groupby(["BatterID", "game_year", "season_half"]).agg(
        nHardHit=("HardHit", "sum"),
        nAir=("Air", "sum"),
        nSwSp=("SwSp", "sum"),
        nPull=("Pull", "sum"),
        nPullAir=("PullAir", "sum"),
        # A feature numerator has to be summable, and neither a mean nor a quantile is,
        # so both exit-speed features are carried multiplied back up by the ball count.
        # The batted-ball row count equals nBIP exactly, so sumEV is the true sum.
        # sumEV90 is an approximation: pooled, it is a BIP-weighted mean of 90th
        # percentiles rather than the 90th percentile of the pooled balls.
        sumEV=("launch_speed", "sum"),
        sumEV90=("launch_speed", lambda x: x.quantile(0.90) * len(x)),
    ).reset_index()

    # Create pitch-level variables
    sc = sc[sc["plate_x"].notna() & sc["plate_z"].notna()] # Functionally removes automatic balls (intentional walks)

    sc["swing"] = np.where(sc["description"].isin(["swinging_strike", "foul", "hit_into_play",
                                                "foul_tip", "foul_bunt", "swinging_strike_blocked",
                                                "missed_bunt", "bunt_foul_tip",
                                                "foul_pitchout", "swinging_pitchout"]), 1, 0)
    sc["whiff"] = np.where(sc["description"].isin(["swinging_strike", "swinging_strike_blocked",
                                                   "foul_tip", "bunt_foul_tip",
                                                   "missed_bunt", "swinging_pitchout"]), 1, 0)
    sc["in_zone"] = np.where((abs(sc["plate_x"]) <= 0.83) & 
                            (sc["plate_z"] >= sc["sz_bot"] - 0.12) & 
                            (sc["plate_z"] <= sc["sz_top"] + 0.12), 
                            1, 0)
    sc["out_zone"] = 1 - sc["in_zone"]

    sc["z_swing"] = np.where(sc["swing"] & sc["in_zone"], 1, 0)
    sc["o_swing"] = np.where(sc["swing"] & sc["out_zone"], 1, 0)
    sc["z_whiff"] = np.where(sc["whiff"] & sc["in_zone"], 1, 0)
    sc["o_whiff"] = np.where(sc["whiff"] & sc["out_zone"], 1, 0)

    # Count the pitch-level features by player-half
    sc = sc.groupby(["BatterID", "game_year", "season_half"]).agg(
        nPitches=("pitch_type", "count"),
        nSwings=("swing", "sum"),
        nWhiffs=("whiff", "sum"),
        nZSwing=("z_swing", "sum"),
        nOSwing=("o_swing", "sum"),
        nInZone=("in_zone", "sum"),
        nOutZone=("out_zone", "sum"),
        nZWhiff=("z_whiff", "sum"),
        nOWhiff=("o_whiff", "sum")
    )


    # Combine back into one player-half panel.
    ph = pa_counts.merge(bb, how="left", on=["BatterID", "game_year", "season_half"])\
                  .merge(sc, how="left", on=["BatterID", "game_year", "season_half"])

    ph[BIP_SUMS] = ph[BIP_SUMS].fillna(0.0)

    ph["SeasonID"] = season_id(ph["game_year"], ph["season_half"])
    return ph


def holdout_players(): 
    """The BatterIDs to project for 2026 1H.

    Returns:
        players (Index): named BatterID, the hitters to project
    """

    holdout = batting_stats_range('2026-03-01', '2026-06-30')
    return pd.Index(holdout["mlbID"].unique(), name="BatterID")


def build_people(statcast_df, players): 
    """Player name and handedness for every batter in the dataset, indexed by BatterID.

    Note: a holdout hitter with no prior half has no panel row at all, and
    would otherwise lose his name in the one table meant to account for him. So names come
    from the Chadwick register by MLBAM id, which covers every requested hitter.

    Bats comes from Statcast's `stand`: a hitter seen from both sides is a switch hitter.
    A hitter with no Statcast pitch -- the holdout hitters with no history -- gets NaN.

    Args:
        statcast_df (DataFrame): the frame load_statcast returns
        players (Index): BatterIDs to include beyond those in statcast_df, e.g. the 2026 1Hholdout
    Returns:
        people (DataFrame): PlayerName and Bats, one row per BatterID
    """

    sides = statcast_df.groupby("BatterID")["stand"].unique()
    bats = sides.map(lambda s: s[0] if len(s) == 1 else "S").rename("Bats")

    ids = pd.Index(bats.index).union(pd.Index(players)).rename("BatterID")
    names = playerid_reverse_lookup(ids.tolist(), key_type="mlbam")
    names = (names.assign(PlayerName=(names["name_first"] + " " + names["name_last"]).str.title())
                  .drop_duplicates("key_mlbam").set_index("key_mlbam")["PlayerName"])

    return pd.DataFrame({"PlayerName": names, "Bats": bats}).reindex(ids)
