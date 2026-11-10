# tests/test_evaluate.py
import numpy as np

from src import config as C
from src.evaluate import my_photos_table

COLUMNS = ["미션", "파일", "통과", "잡힌것", "점수"]


class _Input:
    name = "images"


class FakeSession:
    """첫 클래스를 틀 가운데 크게 잡았다고 답하는 가짜 ONNX 세션."""
    def get_inputs(self):
        return [_Input()]

    def run(self, _, feeds):
        out = np.zeros((1, 4 + len(C.CLASSES), 1), np.float32)
        out[0, :4, 0] = [160, 160, 300, 300]
        out[0, 4 + C.CLASSES[0].id, 0] = 0.9
        return [out]


def test_my_photos_table_empty_has_columns(tmp_path):
    df = my_photos_table(FakeSession(), tmp_path)
    assert df.empty and list(df.columns) == COLUMNS


def test_my_photos_table_uppercase_suffix_and_unreadable_skipped(tmp_path, capsys):
    import cv2
    d = tmp_path / C.CLASSES[0].key
    d.mkdir()
    cv2.imwrite(str(d / "a.png"), np.zeros((400, 600, 3), np.uint8))
    (d / "a.png").rename(d / "A.PNG")
    (d / "broken.jpg").write_bytes(b"not an image")
    df = my_photos_table(FakeSession(), tmp_path)
    assert list(df["파일"]) == ["A.PNG"]
    assert bool(df["통과"].iloc[0])
    assert "broken.jpg" in capsys.readouterr().out
