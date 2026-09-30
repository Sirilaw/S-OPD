import numpy as np
from PIL import Image

from verl.experimental.agent_loop.single_turn_agent_loop import _build_student_teacher_image_views
from verl.workers.config import DistillationConfig


def _image_data(value: int = 128):
    pixels = np.full((6, 7, 3), value, dtype=np.uint8)
    return {"images": [Image.fromarray(pixels, mode="RGB")], "tag": "clean"}


def test_student_noisy_teacher_clean_keeps_teacher_clean_and_student_noisy():
    clean = _image_data()

    student, teacher = _build_student_teacher_image_views(
        clean,
        None,
        enabled=True,
        std=0.2,
        validate=False,
    )

    assert teacher is clean
    assert student is not clean
    np.testing.assert_array_equal(np.asarray(teacher["images"][0]), np.asarray(clean["images"][0]))
    assert not np.array_equal(np.asarray(student["images"][0]), np.asarray(clean["images"][0]))


def test_student_noisy_teacher_clean_preserves_explicit_teacher_view():
    clean = _image_data()
    privileged_teacher = _image_data(value=64)

    _, teacher = _build_student_teacher_image_views(
        clean,
        privileged_teacher,
        enabled=True,
        std=0.2,
        validate=False,
    )

    assert teacher is privileged_teacher


def test_student_noisy_teacher_clean_uses_clean_validation_view():
    clean = _image_data()

    student, teacher = _build_student_teacher_image_views(
        clean,
        None,
        enabled=True,
        std=0.2,
        validate=True,
    )

    assert student is clean
    assert teacher is None


def test_student_noisy_teacher_clean_config_rejects_negative_std():
    try:
        DistillationConfig(
            enabled=True,
            student_noisy_teacher_clean_enabled=True,
            student_noisy_teacher_clean_std=-0.1,
        )
    except ValueError as exc:
        assert "student_noisy_teacher_clean_std" in str(exc)
    else:
        raise AssertionError("negative Gaussian std should fail validation")
