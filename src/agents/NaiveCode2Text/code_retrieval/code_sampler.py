from dotenv import load_dotenv
import duckdb

load_dotenv(override=True)


def sample_codes(population_path: str, code_column: str, n_codes: int):
    """
    Sample codes from a population with replacement.
    The population must be located in a parquet file and have a column for codes.
    """
    con = duckdb.connect(database=":memory:")

    query = f"""
        WITH base AS (
            SELECT {code_column}, row_number() OVER () AS rn
            FROM read_parquet('{population_path}')
        ),
        count_rows AS (
            SELECT COUNT(*) AS total FROM base
        ),
        random_indices AS (
            SELECT
                1 + FLOOR(random() * total)::BIGINT AS rn
            FROM count_rows, range({n_codes})
        )
        SELECT b.{code_column}
        FROM base b
        JOIN random_indices r
        ON b.rn = r.rn
        """
    return con.execute(query).fetchnumpy()[code_column]
