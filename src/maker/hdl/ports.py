"""Verilog port extraction and testbench-stub generation.

Pure functions, extracted from VerilogVerifier. ``extract_ports`` prefers
hdlparse and falls back to a regex so a parse failure still yields something
usable; ``generate_stub_testbench`` builds a ready-to-simulate testbench with
$dumpfile/$dumpvars already wired for VCD capture.
"""
import re


def extract_ports(verilog_code):
    module_match = re.search(r'\bmodule\s+(\w+)', verilog_code)
    if not module_match:
        return None, []
    module_name = module_match.group(1)

    try:
        import hdlparse.verilog_parser as vlog
        parser = vlog.VerilogExtractor()
        objs = parser.extract_objects_from_source(verilog_code)
        for obj in objs:
            if obj.name == module_name:
                ports = []
                for p in obj.ports:
                    ports.append((p.mode, p.name, p.data_type or 'wire'))
                return module_name, ports
    except Exception as e:
        print("hdlparse failed, falling back to regex:", e)

    ports = []
    pattern = r'\b(input|output|inout)\s+(?:reg\s+|wire\s+|logic\s+)?(?:\[[^\]]*\]\s+)?([\w\s,]+)'
    for mode, names_str in re.findall(pattern, verilog_code):
        names = [n.strip() for n in names_str.split(',')]
        for name in names:
            if name:
                ports.append((mode, name, 'wire'))
    return module_name, ports


def generate_stub_testbench(module_name, ports):
    inputs = [p for p in ports if p[0] == 'input']
    outputs = [p for p in ports if p[0] in ('output', 'inout')]

    regs_decl = "\n".join(f"  reg {name};" for _, name, _ in inputs)
    wires_decl = "\n".join(f"  wire {name};" for _, name, _ in outputs)
    port_mapping = ", ".join(f".{name}({name})" for _, name, _ in ports)

    clk_stimulus = ""
    if any(name == 'clk' for _, name, _ in ports):
        clk_stimulus = "\n  always #5 clk = ~clk;\n"

    init_stimulus = "\n  initial begin\n"
    if any(name == 'clk' for _, name, _ in ports):
        init_stimulus += "    clk = 0;\n"
    if any(name == 'reset' for _, name, _ in ports):
        init_stimulus += "    reset = 1;\n    #20 reset = 0;\n"
    elif any(name == 'rst' for _, name, _ in ports):
        init_stimulus += "    rst = 1;\n    #20 rst = 0;\n"

    init_stimulus += '    $dumpfile("sim_out.vcd");\n'
    init_stimulus += f'    $dumpvars(0, tb_{module_name});\n'
    init_stimulus += "    #500;\n"
    init_stimulus += "    $finish;\n"
    init_stimulus += "  end\n"

    return f"""`timescale 1ns/1ps

module tb_{module_name};
  // Inputs
{regs_decl}

  // Outputs
{wires_decl}

  // UUT Instance
  {module_name} uut (
    {port_mapping}
  );
{clk_stimulus}{init_stimulus}
endmodule
"""
