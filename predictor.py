import pandas as pd
import sys

sys.path.append("./")
from model import Kronos, KronosTokenizer, KronosPredictor


class Predictor:
    def __init__(self):

        # 1. Load Model and Tokenizer
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

        # 2. Instantiate Predictor
        self.predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        temperature: float = 1,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ):
        """

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing historical data
        x_timestamp : pd.Series
            Timestamp of the first data point in the input data
            eg. pd.Timestamp("2024-06-30 14:40:15")
        y_timestamp : pd.Series
            Timestamp of the first data point in the prediction
        pred_len : int
            Number of time steps to predict
        temperature : float
            Temperature parameter for sampling
        top_p : float
            Top-p parameter for sampling

        Returns
        -------
                             open       high        low      close       volume     amount
        timestamps
        2024-06-30 14:40:15  10.818892  10.824681  10.773193  10.784436  1409.096069  1504391.0
        pred_df.shape
        (1, 6)
        """

        # 4. Make Prediction
        pred_df = self.predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )

        return pred_df


if __name__ == "__main__":
    predictor = Predictor()
    # 3. Prepare Data
    df = pd.read_csv("./examples/data/XSHG_5min_600977.csv")
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    lookback = 400
    pred_len = 1

    x_df = df.loc[: lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[: lookback - 1, "timestamps"]
    y_timestamp = df.loc[lookback : lookback + pred_len - 1, "timestamps"]

    pred_df = predictor.predict(x_df, x_timestamp, y_timestamp, pred_len)
    print(pred_df)
