"""Read STEP member bindings and device-image slot sizes."""
from pathlib import Path
import re
import sys

TEMPLATE = Path(__file__).with_name("step_image_table.rs.in")
ROOT = TEMPLATE.resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/native/prebuilt"))
from fopt_header import header
from unpack import CRATE


def template_source(paths):
    """Expand carried member names and the family registry paths."""
    carried = {path.rsplit('::', 1)[1]: path for path in paths}

    def member(match):
        name = match[1]
        path = CRATE + name if '::' in name else carried[name]
        return f'Member {{ path: "{path}"'

    def family(match):
        name = match[1]
        bucket = int(name.split('_', 1)[0][1:])
        row = CRATE + 'host::step_image::' + name
        return f'Family {{ bucket: {bucket}, name: "{name}", row: "{row}",'

    source = re.sub(r'Member \{ path: "([\w:]+)"', member, TEMPLATE.read_text())
    return re.sub(r'Family \{ name: "(\w+)",', family, source)


def table(source):
    def records(name):
        fields = r'(\w+): (?:"([^"]+)"|&\[([^\]]*)\]|(\d+))'
        for record in re.findall(rf'^    {name} \{{ (.*) \}},$', source, re.M):
            yield {
                key: string or (int(number) if number else [None if v.strip() == "NO_MEMBER" else int(v)
                                                            for v in values.split(",") if v.strip()])
                for key, string, values, number in re.findall(fields, record)
            }

    return list(records("Member")), list(records("Family"))


def slot_sizes(image):
    slots = header(image)["slots"]
    if any(kind != 0 for kind, _ in slots):
        raise ValueError("STEP requires DRAM slots")
    return [size for _, size in slots]


def write_table(output, text, member_image, family_image):
    """Fill in the measured ABI fields in the member and opcode declarations."""
    def member(match):
        h = header(member_image(match[2]))
        return f'{match[1]}, inputs: {len(h["inputs"])}, outputs: &{h["outputs"]} }},'

    def family(match):
        return f'{match[1]}, slots: {len(slot_sizes(family_image(match[2])))}, {match[3]}'

    text = re.sub(r'^(    Member \{ path: "([^"]+)"), inputs: \d+, outputs: &\[[^\]]*\] \},$',
                  member, text, flags=re.M)
    text = re.sub(r'^(    Family \{ .*row: "([^"]+)"), slots: \d+, (ops: .*)$',
                  family, text, flags=re.M)
    output.write_text(text)
