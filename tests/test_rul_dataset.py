import pandas as pd

from src.prediction.data_loader import RULDataset


def test_rul_dataset_shapes_and_length(tmp_path):
    for index in range(1, 4):
        pd.DataFrame(
            {
                "h": [0.1, 0.2, 0.3, 0.4],
                "v": [0.4, 0.3, 0.2, 0.1],
            }
        ).to_csv(tmp_path / f"{index}.csv", index=False)

    dataset = RULDataset(
        data_dir=str(tmp_path),
        seq_length=2,
        window_size=4,
        cwt_widths=[1, 2],
    )

    x, y = dataset[0]

    assert len(dataset) == 2
    assert x.shape == (2, 2, 2, 4)
    assert y.shape == (1,)
    assert 0.0 <= float(y.item()) <= 1.0
