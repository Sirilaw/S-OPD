from evaluation.geometry3k_oracle.logic_forms import (
    LogicFormError,
    parse_logic_form,
    render_diagram_facts,
    render_logic_form,
)


def test_parse_and_canonicalize_nested_logic_form():
    parsed = parse_logic_form(" Equals( LengthOf(Line(A, D)), 3 ) ")
    assert parsed.canonical() == "Equals(LengthOf(Line(A,D)),3)"


def test_algebraic_and_latex_leaves_are_preserved():
    assert render_logic_form("Equals(MeasureOf(Angle(A,B,C)),2x+10)").text == (
        "The measure of angle ABC is 2x+10."
    )
    assert render_logic_form(r"Equals(LengthOf(Line(A,C)),3\sqrt{2})").text == (
        r"The length of line AC is 3\sqrt{2}."
    )
    assert render_logic_form("Equals(MeasureOf(Angle(A,B,C)),MeasureOf(angle 13))").text == (
        "Angle ABC and angle 13 have equal measures."
    )


def test_required_rendering_examples():
    assert render_logic_form("Triangle(A,B,C)").text == "A, B, and C form triangle ABC."
    assert render_logic_form("PointLiesOnLine(D,Line(A,B))").text == "D lies on line AB."
    assert render_logic_form("Perpendicular(Line(A,C),Line(B,C))").text == "AC is perpendicular to BC."
    assert render_logic_form("Parallel(Line(A,B),Line(C,D))").text == "AB is parallel to CD."
    assert render_logic_form("Equals(LengthOf(Line(A,D)),3)").text == "The length of line AD is 3."
    assert render_logic_form("Equals(LengthOf(Line(A,B)),LengthOf(Line(A,C)))").text == "AB and AC have equal lengths."


def test_unknown_is_preserved_and_reported():
    rendered = render_logic_form("NewDiagramPredicate(A,B)")
    assert not rendered.supported
    assert rendered.text == "[Logic form: NewDiagramPredicate(A,B)]"


def test_deduplicates_without_reordering():
    facts = render_diagram_facts(["Parallel(Line(A,B),Line(C,D))", "Parallel(Line(A,B), Line(C,D))"])
    assert len(facts) == 1


def test_find_is_rejected_even_if_nested():
    try:
        render_logic_form("Equals(Find(LengthOf(Line(A,B))),3)")
    except LogicFormError:
        pass
    else:
        raise AssertionError("Find must never enter oracle facts")


def test_blank_logic_form_is_skipped_but_not_rendered_into_prompt():
    facts = render_diagram_facts(["", "Parallel(Line(A,B),Line(C,D))"])
    assert [fact.text for fact in facts] == ["AB is parallel to CD."]
