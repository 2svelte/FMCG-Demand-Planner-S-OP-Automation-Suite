import pandas as pd
import pytest

from omniplan import data_gen, forecast, master_data, scenario
from omniplan.excel_export import build_workbook
from omniplan.loaders import clean_master, clean_pos, clean_promo


@pytest.fixture(scope="session")
def world():
    pos, master, promo = data_gen.generate()
    pos, master, promo = clean_pos(pos), clean_master(master), clean_promo(promo)
    base = forecast.fit_baselines(pos, 12)
    lift = forecast.promo_lift_table(pos)
    return pos, master, promo, base, lift
