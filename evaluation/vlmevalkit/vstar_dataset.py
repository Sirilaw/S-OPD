"""V* Bench adapter that guarantees judge-free exact MCQ scoring."""

from vlmeval.dataset.image_mcq import ImageMCQDataset


class VStarBenchDataset(ImageMCQDataset):
    """Use VLMEvalKit's VStarBench data with deterministic option scoring."""

    def evaluate(self, eval_file, **judge_kwargs):
        judge_kwargs = dict(judge_kwargs)
        judge_kwargs["model"] = "exact_matching"
        return self.evaluate_heuristic(eval_file, **judge_kwargs)
