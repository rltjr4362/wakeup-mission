# tests/test_data.py
from src import config as C
from src.data import dedupe_rows, dedupe_labels, label_counts


def test_dedupe_rows_drops_same_class_overlap_keeps_order():
    a = (8, 0.5, 0.5, 0.4, 0.4)
    b = (8, 0.51, 0.5, 0.4, 0.4)      # a와 거의 같음 → 제거
    c = (9, 0.51, 0.5, 0.4, 0.4)      # 다른 클래스 → 유지
    d = (8, 0.1, 0.1, 0.1, 0.1)       # 안 겹침 → 유지
    assert dedupe_rows([a, b, c, d]) == [a, c, d]


def test_dedupe_rows_keeps_low_iou():
    a = (0, 0.3, 0.5, 0.4, 0.4)
    b = (0, 0.5, 0.5, 0.4, 0.4)       # IoU = 1/3 → 유지
    assert dedupe_rows([a, b]) == [a, b]


def test_dedupe_labels_rewrites_files(tmp_path):
    lab = tmp_path / "labels" / "train"
    lab.mkdir(parents=True)
    (lab / "a.txt").write_text("8 0.5 0.5 0.4 0.4\n8 0.51 0.5 0.4 0.4\n9 0.2 0.2 0.1 0.1\n")
    removed = dedupe_labels(tmp_path, "train")
    assert removed == 1
    assert (lab / "a.txt").read_text() == "8 0.500000 0.500000 0.400000 0.400000\n9 0.200000 0.200000 0.100000 0.100000\n"


def test_label_counts_reads_label_files(tmp_path):
    lab = tmp_path / "labels" / "val"
    lab.mkdir(parents=True)
    (lab / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n0 0.1 0.1 0.1 0.1\n3 0.2 0.2 0.1 0.1\n")
    (lab / "b.txt").write_text("3 0.5 0.5 0.4 0.4\n\n")
    (lab / "c.txt").write_text("")
    df = label_counts(tmp_path, "val").set_index("id")
    assert list(df.columns) == ["사진", "박스"]
    assert len(df) == len(C.CLASSES)
    assert df.loc[0].tolist() == [1, 2]
    assert df.loc[3].tolist() == [2, 2]
    assert df.loc[1].tolist() == [0, 0]


def test_per_class_counts_int_and_dict():
    from src.data import per_class_counts
    assert per_class_counts(400, C.CLASSES13) == {c.key: 400 for c in C.CLASSES13}
    got = per_class_counts({"*": 400, "door": 1000}, C.CLASSES13)
    assert got["door"] == 1000 and got["cup"] == 400 and len(got) == 13


def test_label_counts_13_classes(tmp_path):
    lab = tmp_path / "labels" / "val"
    lab.mkdir(parents=True)
    (lab / "a.txt").write_text("12 0.5 0.5 0.4 0.4\n")
    df = label_counts(tmp_path, "val", C.CLASSES13).set_index("id")
    assert len(df) == 13 and df.loc[12].tolist() == [1, 1]
