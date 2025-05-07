import pandas as pd
import requests
from io import StringIO

def load_data_from_url(url: str) -> pd.DataFrame:
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

def validate_data(df: pd.DataFrame) -> bool:
    # Ожидаем колонки: "Password", "Times"
    return "Password" in df.columns and "Times" in df.columns and not df.empty
