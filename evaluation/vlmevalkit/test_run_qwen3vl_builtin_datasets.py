from run_qwen3vl import builtin_dataset_class


def test_hallucination_benchmarks_use_yes_no_dataset_adapter():
    assert builtin_dataset_class("POPE") == "ImageYORNDataset"
    assert builtin_dataset_class("HallusionBench") == "ImageYORNDataset"


def test_unknown_builtin_benchmark_keeps_mcq_fallback():
    assert builtin_dataset_class("SomeFutureMCQ") == "ImageMCQDataset"


def test_vstar_uses_mcq_and_zoombench_uses_dual_view_adapter():
    assert builtin_dataset_class("VStarBench") == "VStarBenchDataset"
    assert builtin_dataset_class("ZoomBench") == "ZoomBenchDataset"
    assert builtin_dataset_class("ZoomBench_Crop") == "ZoomBenchDataset"
