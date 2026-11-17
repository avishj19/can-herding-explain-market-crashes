"""Tests for scripts/party_figures.py (Figs 13-15): every figure is drawn from small
made-up returns, and the numbers it reports agree with the same numbers computed here
in a different way (parties from hand-typed dates, plain numpy sums)."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.image import imread

import party_figures as pf
from herding import politics


def market(seed=0, start="1989-06-01", end="2012-12-31"):
    """Made-up daily log returns over Bush (R), Clinton (D), Bush (R), Obama (D)."""
    days = pd.bdate_range(start, end)
    return pd.Series(0.01 * np.random.default_rng(seed).standard_t(4, len(days)) + 0.0003, index=days)


def dem_by_hand(dates):
    """Democratic days from typed-in inauguration dates (not politics.democrat)."""
    d = pd.DatetimeIndex(dates)
    return ((d >= "1993-01-20") & (d < "2001-01-20")) | ((d >= "2009-01-20") & (d < "2017-01-20"))


def is_image(path):
    return path.exists() and imread(path).shape[0] > 200


# ---------------------------------------------------------------- small helpers
def test_years_between_counts_both_end_days():
    assert pf.years_between("2001-01-01", "2001-12-31") == pytest.approx(365 / 365.25)
    assert pf.years_between("2000-03-01", "2000-03-01") == pytest.approx(1 / 365.25)


def test_event_name_only_inside_the_windows():
    assert pf.event_name("1987-10-19") == "Black Monday"
    assert pf.event_name("1987-10-20") == ""               # one day late: no label
    assert pf.event_name("1929-10-28") == "1929 crash"
    assert pf.event_name("2008-10-15") == "2008 financial crisis"
    assert pf.event_name("2020-03-16") == "Covid panic"
    assert pf.event_name("1962-05-28") == ""


def test_event_days_finds_the_biggest_fall_and_skips_events_outside_the_data():
    r = market()
    r.loc["2008-11-20"] = -0.5
    assert pf.event_days(r) == [(pd.Timestamp("2008-11-20"), "2008 financial crisis")]
    assert pf.event_days(r.loc["1995":"2000"]) == []


def test_money_is_short_and_keeps_the_dollar_sign_literal():
    assert pf.money(0.853) == r"\$0.85"
    assert pf.money(9.99) == r"\$9.99"
    assert pf.money(1639.4) == r"\$1,639"


def test_term_table_matches_numbers_worked_out_by_hand():
    r = market(1)
    table = pf.term_table(r)
    assert [t["name"] for t in table] == ["Bush", "Clinton", "Bush", "Obama"]
    assert [t["party"] for t in table] == ["R", "D", "R", "D"]
    clinton = r.loc["1993-01-20":"2001-01-19"]
    years = ((clinton.index[-1] - clinton.index[0]).days + 1) / 365.25
    t = table[1]
    assert t["label"] == "Clinton (D)  1993–01"
    assert t["years"] == pytest.approx(years)
    assert t["yearly"] == pytest.approx(np.exp(clinton.sum()) ** (1 / years) - 1)
    assert t["volatility"] == pytest.approx(clinton.std() * np.sqrt(len(clinton) / years))
    assert t["worst"] == pytest.approx(np.exp(clinton.min()) - 1)
    assert t["worst_date"] == clinton.idxmin()
    assert sum(t["total"] for t in table) == pytest.approx(r.sum())      # no day lost or counted twice


def test_term_table_on_a_known_tiny_market():
    # 2% up every day for a term: the yearly return is fixed by the day count alone
    days = pd.bdate_range("1993-01-20", "2001-01-19")
    t, = pf.term_table(pd.Series(np.log(1.02), index=days))
    assert t["worst"] == pytest.approx(0.02) and t["volatility"] == pytest.approx(0, abs=1e-9)
    assert (1 + t["yearly"]) ** t["years"] == pytest.approx(1.02 ** len(days))


def test_term_table_leaves_out_terms_with_too_few_days():
    r = market(2, start="2000-12-20")           # about a month of Clinton
    assert [t["name"] for t in pf.term_table(r, min_days=30)] == ["Bush", "Obama"]
    assert [t["name"] for t in pf.term_table(r, min_days=1)] == ["Clinton", "Bush", "Obama"]


def test_dollar_under_moves_only_on_that_partys_days():
    r = np.log([1.10, 1.20, 0.50, 2.00])
    dem = [True, False, False, True]
    assert pf.dollar_under(r, dem, "D") == pytest.approx([1.10, 1.10, 1.10, 2.20])
    assert pf.dollar_under(r, dem, "R") == pytest.approx([1.00, 1.20, 0.60, 0.60])
    both = pf.dollar_under(r, dem, "D")[-1] * pf.dollar_under(r, dem, "R")[-1]
    assert both == pytest.approx(np.exp(r.sum()))            # the two halves make the whole


def test_gap_percent_is_the_difference_of_the_two_averages():
    r = market(3)
    dem = dem_by_hand(r.index)
    by_hand = 100 * 252 * (r.values[dem].mean() - r.values[~dem].mean())
    assert pf.gap_percent(r.values, dem) == pytest.approx(by_hand)
    assert pf.gap_percent(r.values, ~dem) == pytest.approx(-by_hand)     # swapping parties flips the sign


def test_gap_percent_of_a_planted_gap():
    days = pd.bdate_range("1989-06-01", "2012-12-31")
    dem = dem_by_hand(days)
    r = np.where(dem, 0.001, -0.001) + np.tile([0.01, -0.01], len(days))[:len(days)]
    assert pf.gap_percent(r, dem) == pytest.approx(100 * 252 * 0.002, rel=0.02)


# ---------------------------------------------------------------- Fig 13
def test_party_totals_match_plain_sums():
    r = market(4)
    dem = dem_by_hand(r.index)
    tot = pf.party_totals(r)
    assert tot["D"]["dollar"] == pytest.approx(np.exp(r.values[dem].sum()))
    assert tot["R"]["dollar"] == pytest.approx(np.exp(r.values[~dem].sum()))
    assert tot["D"]["years"] + tot["R"]["years"] == pytest.approx(
        ((r.index[-1] - r.index[0]).days + 1) / 365.25, abs=0.03)       # weekends at the hand-overs
    assert tot["D"]["yearly"] == pytest.approx(tot["D"]["dollar"] ** (1 / tot["D"]["years"]) - 1)


def test_party_totals_leave_out_the_most_extreme_term():
    r = market(5)
    r.loc["1993-01-20":"2001-01-19"] += 0.002          # make Clinton the best Democratic term
    r.loc["2001-01-20":"2009-01-19"] -= 0.002          # and Bush 2001 the worst Republican one
    tot = pf.party_totals(r)
    assert (tot["D"]["extreme"], tot["D"]["extreme_span"]) == ("Clinton", "1993–01")
    assert (tot["R"]["extreme"], tot["R"]["extreme_span"]) == ("Bush", "2001–09")
    assert tot["D"]["dollar_without"] == pytest.approx(np.exp(r.loc["2009-01-20":].sum()))
    assert tot["R"]["dollar_without"] == pytest.approx(np.exp(r.loc[:"1993-01-19"].sum()))


def test_fig13_is_drawn_and_reports_the_right_dollars(tmp_path):
    r = market(6)
    dem = dem_by_hand(r.index)
    out = pf.fig13_long_view(r, tmp_path / "f13.png")
    assert is_image(tmp_path / "f13.png")
    assert out["final"] == pytest.approx(np.exp(r.sum()))
    assert out["dollar_D"] == pytest.approx(np.exp(r.values[dem].sum()))
    assert out["dollar_R"] == pytest.approx(np.exp(r.values[~dem].sum()))
    assert out["dollar_D"] * out["dollar_R"] == pytest.approx(out["final"])


# ---------------------------------------------------------------- Fig 14
def test_fig14_is_drawn_with_one_row_per_presidency(tmp_path):
    r = market(7)
    r.loc["1997-10-27"] = np.log(0.7)                  # a planted 30% fall under Clinton
    table = pf.fig14_every_presidency(r, tmp_path / "f14.png")
    assert is_image(tmp_path / "f14.png")
    assert [t["label"] for t in table] == ["Bush (R)  1989–93", "Clinton (D)  1993–01",
                                           "Bush (R)  2001–09", "Obama (D)  2009–12"]
    assert table[1]["worst"] == pytest.approx(-0.30)
    assert table[1]["worst_date"] == pd.Timestamp("1997-10-27")
    obama = r.loc["2009-01-20":]
    assert table[3]["volatility"] == pytest.approx(
        obama.std() * np.sqrt(len(obama) / (((obama.index[-1] - obama.index[0]).days + 1) / 365.25)))


# ---------------------------------------------------------------- Fig 15
def test_luck_test_matches_gaps_rolled_by_hand():
    r = market(8)
    dem = dem_by_hand(r.index)
    test = pf.luck_test(r)
    gap = lambda d: 100 * 252 * (r.values[d].mean() - r.values[~d].mean())
    shifts = range(252, len(r) - 252 + 1, 63)
    by_hand = np.array([gap(np.roll(dem, k)) for k in shifts])
    assert test["real"] == pytest.approx(gap(dem))
    assert test["fake"] == pytest.approx(by_hand)
    assert test["percentile"] == pytest.approx(100 * np.mean(by_hand < gap(dem)))


def test_luck_test_puts_a_planted_gap_at_the_top():
    r = market(9)
    before = pf.luck_test(r)["real"]
    r[dem_by_hand(r.index)] += 0.003                   # Democrats get +0.3% a day
    test = pf.luck_test(r)
    assert test["percentile"] == 100
    # the gap is a difference of means, so the planted 0.3% a day adds exactly
    # 100 * 252 * 0.003 = 75.6% a year. (The old check, real > 60, ignored the luck in the
    # gap itself, about 9% a year: it failed on 1 of 39 other seeds, overnight stress test.)
    assert test["real"] - before == pytest.approx(75.6)


def test_rolling_gaps_match_windows_cut_by_hand():
    r = market(10)
    rolling = dict(pf.rolling_gaps(r, window_years=10))
    assert list(rolling) == list(range(1999, 2013))
    x = r.loc["1995-01-01":"2004-12-31"]
    dem = dem_by_hand(x.index)
    assert rolling[2004] == pytest.approx(100 * 252 * (x.values[dem].mean() - x.values[~dem].mean()))


def test_rolling_gaps_skip_windows_with_one_party_nearly_missing():
    r = market(11)
    # 4-year windows ending 1996 (1993-96) hold only about 13 Bush days: skipped
    years = [y for y, _ in pf.rolling_gaps(r, window_years=4)]
    assert 1996 not in years and 1994 in years
    assert pf.rolling_gaps(r, window_years=40) == []   # longer than the data: nothing to show


def test_fig15_is_drawn_and_reports_each_market(tmp_path):
    markets = {"market A": market(12), "market B": market(13, start="1991-01-01")}
    out = pf.fig15_luck(markets, tmp_path / "f15.png", window_years=10)
    assert is_image(tmp_path / "f15.png")
    assert set(out["tests"]) == set(markets)
    for name, r in markets.items():
        dem = dem_by_hand(r.index)
        assert out["tests"][name]["real"] == pytest.approx(
            100 * 252 * (r.values[dem].mean() - r.values[~dem].mean()))
        assert 0 <= out["tests"][name]["percentile"] <= 100
    assert out["rolling"] == pf.rolling_gaps(markets["market A"], 10)


def test_party_colours_are_fig12s():
    import make_figures as mf
    assert pf.SHADE == {"D": mf.DEM_SHADE, "R": mf.REP_SHADE}
    assert politics.PRESIDENTS[0][0] <= "1896-05-27"          # the table covers the whole Dow


@pytest.mark.slow
def test_main_writes_the_three_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "FIG", tmp_path)
    saved = {}
    for name in ("fig13_long_view", "fig14_every_presidency"):
        real = getattr(pf, name)
        monkeypatch.setattr(pf, name, lambda r, real=real, name=name: saved.setdefault(name, real(r, tmp_path / f"{name}.png")))
    real15 = pf.fig15_luck
    monkeypatch.setattr(pf, "fig15_luck", lambda m: real15(m, tmp_path / "fig15.png"))
    pf.main()
    assert len(list(tmp_path.glob("*.png"))) == 3
    assert saved["fig13_long_view"]["years_D"] + saved["fig13_long_view"]["years_R"] == pytest.approx(130.4, abs=0.2)
