from predictor import Predictor
from fastapi import FastAPI, HTTPException, APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

import uvicorn
from functools import wraps
import math
import time
import datetime
from pathlib import Path
import yaml
import json
import inspect
from fastapi.testclient import TestClient
import sys

sys.path.append(".")


app = FastAPI(
    title="QMT Data Service", description="API wrapper for XTData functions", version="1.0.0"
)

predictor = Predictor()


def dict_to_df(data_dict: Dict) -> pd.DataFrame:
    """Convert dict to pandas DataFrame

    Parameters
    ----------
    data : Dict
        Serialized DataFrame

    Returns
    -------
    df pd.DataFrame
        Deserialized DataFrames
    """
    df = pd.DataFrame(data_dict["data"], index=data_dict["index"], columns=data_dict["columns"])
    return df


def serialize_data(
    data: Union[pd.DataFrame, Dict, List, float, object],
) -> Union[Dict, List, None, float]:
    """Convert various objects to JSON-serializable format with class support
    通过data.__class__.__name__.startswith('Xt') 支持对qmt内部类的序列化。遇到这种类会依次遍历成员变量然后构成dict
    """
    if isinstance(data, pd.DataFrame):
        return {
            "index": [serialize_data(element) for element in data.index],
            "columns": data.columns.tolist(),
            "data": [[serialize_data(element) for element in row] for row in data.values.tolist()],
        }
    elif isinstance(data, pd.Series):
        return [serialize_data(element) for element in data.tolist()]
    elif isinstance(data, pd.Timestamp):
        return data.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(data, dict):
        return {k: serialize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_data(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, float):
        return None if math.isnan(data) else data
    elif data.__class__.__name__.startswith("Xt"):
        result = {}
        for name, value in inspect.getmembers(data):
            # 过滤掉魔法方法和类方法
            if not name.startswith("__") and not inspect.ismethod(value):
                result[name] = serialize_data(value)
        return result
    else:
        return data


class PredictRequest(BaseModel):
    sequential_data: Dict
    x_timestamp: List[str]
    y_timestamp: List[str]
    pred_len: int
    temperature: float = 1
    top_p: float = 0.9
    sample_count: int = 1
    verbose: bool = True


@app.post("/llm/o/precict", response_model=Dict[str, List])
async def predict(req: PredictRequest) -> Dict[str, List]:
    """Predict using the predictor

    Parameters
    ----------
    req : PredictRequest
        keyword arguments for predictor.predict
        sequential_data : Dict
            sequential_data is convert by ori_data(dataFrame) containing historical data.
            "index": [convert(element) for element in ori_data.index],
            "columns": ori_data.columns.tolist(),
            "data": [[convert(element) for element in row] for row in ori_data.values.tolist()],
        x_timestamp : List[str]
            Timestamp of the sequential_data
            eg. [..., "2024-06-30 15:00:00", "2024-07-01 15:00:00"]
        y_timestamp : pd.Series
            Timestamp to be predict
            eg. ["2024-07-02 15:00:00", "2024-07-03 15:00:00", ...]
        pred_len : int
            Number of time steps to predict, same as the length of y_timestamp
        temperature : float
            Temperature parameter for sampling
        top_p : float
            Top-p parameter for sampling
        sample_count : int
            Number of samples to generate
        verbose : bool
            Whether to print progress

    Returns
    -------
    predict_result : Dict
        predict_result is convert by predict_result_df(dataFrame) containing historical data.
        "index": [convert(element) for element in predict_result_df.index],
        "columns": predict_result_df.columns.tolist(),
        "data": [[convert(element) for element in row] for row in predict_result_df.values.tolist()],
        eg.
        {
            "index": ["2024-07-02 15:00:00", "2024-07-03 15:00:00", ...],
            "columns": ["open", "high", "low", "close", "volume", "amount"],
            "data": [[10.818892, 10.824681, 10.773193, 10.784436, 1409.096069, 1504391.0],
                    [10.818892, 10.824681, 10.773193, 10.784436, 1409.096069, 1504391.0], ...]
        }
    """
    # return data.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")

    df = dict_to_df(req.sequential_data)
    x_timestamp = pd.to_datetime(req.x_timestamp).to_series()
    y_timestamp = pd.to_datetime(req.y_timestamp).to_series()
    pred_len = req.pred_len
    temperature = req.temperature
    top_p = req.top_p
    sample_count = req.sample_count
    verbose = req.verbose
    # print(df)
    # print(x_timestamp)
    predict_result = predictor.predict(
        df, x_timestamp, y_timestamp, pred_len, temperature, top_p, sample_count, verbose
    )
    print(predict_result)
    predict_result = serialize_data(predict_result)
    return predict_result


def test():
    client = TestClient(app)

    date_range = pd.date_range(start="2024-06-01 15:00:00", end="2024-07-01 15:00:00", freq="D")
    data = {
        "open": np.random.rand(len(date_range)),
        "high": np.random.rand(len(date_range)),
        "low": np.random.rand(len(date_range)),
        "close": np.random.rand(len(date_range)),
        "volume": np.random.randint(1000, 10000, len(date_range)),
        "amount": np.random.randint(10000, 100000, len(date_range)),
    }
    sequential_data_df = pd.DataFrame(data)

    # 设置时间戳
    x_timestamp = date_range.strftime("%Y-%m-%d %H:%M:%S").tolist()
    y_timestamp = ["2024-07-02 15:00:00"]
    pred_len = len(y_timestamp)
    sequential_data = serialize_data(sequential_data_df)

    data = {
        "sequential_data": sequential_data,
        "x_timestamp": x_timestamp,
        "y_timestamp": y_timestamp,
        "pred_len": pred_len,
    }
    headers = {"Content-Type": "application/json"}
    response = client.post(
        "/llm/o/precict",  # 添加请求路径
        json=data,  # 使用json参数自动序列化
        headers=headers,
    )
    predict_result = dict_to_df(response.json())
    print("predict result:", predict_result)


if __name__ == "__main__":
    test()
    # uvicorn.run(app, host="0.0.0.0", port=8000)
