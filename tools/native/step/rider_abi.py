"""Production RSS frames; every alias is verified against bound addresses at run time.

Dense recipes use the planner's adjacent off/cap tables and a packed, constant
group-row tensor. Sparse recipes retain each group's independent bindings.
Unread slots are removed only after scanning the captured member task C.
"""
import re


def roles(groups, layer):
    group = [f'g{g}.{s}' for g in range(groups) for s in ('off', 'cap', 'rows')]
    common = ['out', 'rows', 'stage', 'publish', 'rope', 'x']
    const = ['const.h64', 'const.h60', 'const.eye', 'const.h4', 'const.h8']
    o = ['o.sigma', 'o.wo', 'o.ws', 'o.signs', 'o.post']
    f = ['f.sigma_down', 'f.sigma', 'f.pre', 'f.wug', 'f.ws', 'f.wd', 'f.ds', 'f.post',
         'f.pre_signed', 'f.codes', 'f.scales']
    n = ['n.sigma', 'n.in', 'n.wq', 'n.ws', 'n.q', 'n.k']
    weights = [f'L{layer}.{s}' for s in o + f + ['ls'] + n]
    pools = ['cos', 'sin', f'k{layer}', f'v{layer}', 'qin', f'k{layer+1}', f'v{layer+1}']
    prompt = [f'p{r}.{s}' for r in range(3) for s in ('off', 'tm', 'cap')] + ['tm_rows', 'cap_rows', 'cap_hi', 'out']
    return group + common + const + weights + pools + prompt


def used_slots(source, count):
    assert source.count('#define DRAM_IO(slot) (DRAM_IO_BEGIN + (slot))') == 1
    assert not re.search(r'addrs\[\d+\]', source), 'unrecognized direct argument access'
    uses = set(int(s) for s in re.findall(r'DRAM_IO\((\d+)\)', source))
    assert uses and max(uses) < count
    # Only numeric uses and the two compiler macro definitions are allowed.
    assert source.count('#define SRAM_IO(slot) DRAM_IO(slot)') == 1
    assert source.count('DRAM_IO(') == len(re.findall(r'DRAM_IO\(\d+\)', source)) + 2
    return sorted(uses)


def recipe(groups, sizes, used, layers, dense):
    names = roles(groups, 0)
    assert len(names) == len(sizes), (len(names), len(sizes))
    origins, indices, mapping = [], {}, []
    for layer in range(layers):
        row = [[-1, 0] for _ in names]
        for slot, name in enumerate(roles(groups, layer)):
            if slot not in used:
                continue
            offset = 0
            if dense and slot < 3 * groups:
                g = slot // 3
                # Paired groups pack off/cap/rows triples in one upload.
                # The last triple may be a single eight-row group.
                assert sizes[3*g] == sizes[3*g+1]
                assert sizes[3*g+2] in (64, 128)
                assert sizes[3*g] in (sizes[3*g+2] * 128, sizes[3*g+2] * 192)
                name, offset = 'decode.paired_words', sum(sizes[:slot])
            if name not in indices:
                assert offset == 0, (name, offset)
                indices[name] = len(origins)
                origins.append([layer, slot])
            row[slot] = [indices[name], offset]
        mapping.append(row)
    return dict(count=layers, dense=dense, slots=len(origins), origins=origins,
                mapping=mapping, used=used, sizes=sizes)


def paired_recipes(decode_groups, sizes, used, first_opcode):
    groups = (decode_groups + 1) // 2
    out = []
    for dense in (False, True):
        for n in range(2, 5):
            r = recipe(groups, sizes, used, n, dense)
            if r['slots'] + 1 <= 115:
                r['opcode'] = first_opcode + len(out)
                assert r['opcode'] < 32
                out.append(r)
    assert any(r['count'] == 2 and not r['dense'] for r in out)
    assert any(r['count'] == 4 for r in out)
    return out
