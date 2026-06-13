from grid_loader import load_network, run_power_flow
from violation_detector import detect_violations


def test_detects_forced_line_overload():
    net = load_network("IEEE 14-bus")
    net.line["max_loading_percent"] = 1.0

    assert run_power_flow(net)
    report = detect_violations(net)

    assert report.converged
    assert len(report.overloaded_lines) > 0


def test_dc_power_flow_does_not_report_voltage_magnitude_violations():
    net = load_network("IEEE 14-bus")
    net.bus["min_vm_pu"] = 1.10

    assert run_power_flow(net, mode="dc")
    report = detect_violations(net, include_voltage=False)

    assert report.converged
    assert len(report.low_voltage_buses) == 0
    assert len(report.high_voltage_buses) == 0


def test_ac_power_flow_can_report_voltage_magnitude_violations():
    net = load_network("IEEE 14-bus")
    net.bus["min_vm_pu"] = 1.10

    assert run_power_flow(net, mode="ac")
    report = detect_violations(net, include_voltage=True)

    assert report.converged
    assert len(report.low_voltage_buses) > 0
