from dotenv import load_dotenv
import pandas as pd

load_dotenv(override=True)


def sample_codes(population: pd.Dataframe, n_codes: int, replace: bool):
    return population.sample(n_codes, replace=replace)["CODE"].to_numpy()


def connect_code_to_notice(code):
    # TODO
    return None
