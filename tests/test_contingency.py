from contingency import apply_contingency, list_contingencies
from grid_loader import load_network


def test_apply_line_contingency_sets_line_out_of_service():
    net = load_network("IEEE 14-bus")
    contingency = next(item for item in list_contingencies(net) if item.kind == "line")

    outaged = apply_contingency(net, contingency)

    assert bool(net.line.at[contingency.element_index, "in_service"])
    assert not bool(outaged.line.at[contingency.element_index, "in_service"])


def test_apply_generator_contingency_sets_generator_out_of_service():
    net = load_network("IEEE 14-bus")
    contingency = next(item for item in list_contingencies(net) if item.kind == "generator")

    outaged = apply_contingency(net, contingency)

    assert bool(net.gen.at[contingency.element_index, "in_service"])
    assert not bool(outaged.gen.at[contingency.element_index, "in_service"])
