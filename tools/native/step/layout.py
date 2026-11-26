"""Generate the STEP dispatcher and link layout from the serving ABI."""
from binding_check import slot_sizes
from rider_abi import paired_recipes, used_slots

FULL_OPS = (4, 6, 7, 8, *range(10, 20))
G12 = 't1024_g12_p941_pair'
# The STEP weight image reuses pages 0, 4 and 5 from last_bf16_out128.
# Page -1 is the generated i8 mask. Preserve the recorded page layout so
# member references and the narrow RSS/RFS weight matching use the same bases.
TABLE_PAGES = [5, 5, 4, *([5] * 6), 4, 5, 0, 0, 0, 0, 4, 5, 5, 5, 4]
LAYOUTS = {
    G12: ({1: 110, 3: 24, 23: 26}, (28, 208), 228, (4, 133, 166)),
    't1536_g12_p999_pair': ({0: 8, 1: 198, 2: 8, 3: 24}, (28, 164), 252, (4, 123)),
}

# Reserve the staging capacity of each released composition.
STAGING = {
    G12: 3475712,
    't1536_g12_p999_pair': 2580224,
}
LINK_FLAGS = ('-flto', '-flto-partition=one')


def layout(name):
    members, tables, length, masks = LAYOUTS[name]
    offsets = members.copy()
    full_base = tables[0] + 40
    full_positions = (0, 2, 4, 6, 8, 10, 14, 16, 18, 20, 22, 24, 28, 30)
    offsets.update((op, full_base + page) for op, page in zip(FULL_OPS, full_positions, strict=True))
    pages = [4] * length
    for index in tables:
        pages[index:index+len(TABLE_PAGES)] = TABLE_PAGES
    for index in masks:
        pages[index:index+4] = [-1] * 4
    return offsets, pages


def metadata(name, slots, weight_len):
    wide = name.startswith('t1536')
    sizes = [4] * (slots - 1) + [8]
    return {
        'chips': 1, 'pes': 8, 'inputs': list(range(slots)), 'outputs': [],
        'slots': [{'kind': 'dram', 'size': size} for size in sizes],
        'weight': {'kind': 'dram', 'len': weight_len},
        'stacks': [{'kind': 'dram', 'size': 56680448 if wide else 37785600},
                   {'kind': 'dram', 'size': 0}],
        'sram_end': 524288, 'staging': STAGING[name],
        'profile': {'depth': {'debug': 32 if wide else 8, 'info': 32 if wide else 8,
                              'trace': 2784 if wide else 696},
                    'spans': [[['Task', 0, 1]], [['Task', 0, 1]]]},
        'tasks': [[0, 0], [0, 0]],
    }


def chunk_opcodes(name, paths):
    resident = [0, 1, 2]
    if name == G12:
        resident += [5, 9, 23]
    attention = [op for op, path in enumerate(paths) if path is not None and op not in resident]
    return resident, attention


def rss_loop(recipe):
    opcode = recipe['opcode']
    count = recipe['count']
    width = len(recipe['sizes'])
    maps = ',\n'.join('{' + ','.join(str(k) for k, _ in row) + '}' for row in recipe['mapping'])
    offsets = ',\n'.join('{' + ','.join(str(off) for _, off in row) + '}' for row in recipe['mapping'])
    loop = f"""static void loop_{opcode}(uint64_t *addrs) {{
    uint64_t frame[{4 + width}];
    for (unsigned j = 0; j < 4; ++j) frame[j] = addrs[j];
    #pragma GCC unroll 1
    for (unsigned i = 0; i < {count}; ++i) {{
        for (unsigned j = 0; j < {width}; ++j) {{
            int k = map_{opcode}[i][j];
            frame[4+j] = k < 0 ? 0 : addrs[4+k] + offset_{opcode}[i][j];
        }}
        dispatch_body_0(frame);
    }}
}}"""
    return (f'static const signed char map_{opcode}[{count}][{width}] = {{{maps}}};\n'
            f'static const unsigned int offset_{opcode}[{count}][{width}] = {{{offsets}}};\n'
            + loop)


def full_sequence(name, start, wide):
    cases = '\n'.join(f'        case {op}: dispatch_body_{op}(frame); break;' for op in FULL_OPS)
    arguments = f'        uint64_t *args = addrs + {start} + 10*i;\n' if wide else ''
    frame = 'args' if wide else 'addrs'
    base = '' if wide else f'{start}+10*i+'
    address = f'{frame}[{base}9]'
    argument = f'{frame}[{base}j]'
    loop = f"""static void {name}(uint64_t *addrs, unsigned n) {{
    uint64_t frame[13];
    for (unsigned j = 0; j < 4; ++j) frame[j] = addrs[j];
    #pragma GCC unroll 1
    for (unsigned i = 0; i < n; ++i) {{
{arguments}        for (unsigned j = 0; j < 9; ++j) frame[4+j] = {argument};
        switch (({address} & 0x1ff) >> 3) {{
{cases}
        default: __builtin_trap();
        }}
    }}
}}"""
    return loop


def rider_variants(name, slots, task, image):
    groups = int(name.split('_')[1][1:])
    sizes = slot_sizes(image)
    return [v for v in paired_recipes(groups, sizes, used_slots(task, len(sizes)), 26)
            if v['slots'] < slots]


def rider_entries(variants):
    for variant in variants:
        checks = [(layer, slot, *row[slot]) for layer, row in enumerate(variant['mapping'])
                  for slot in variant['used']]
        origins = [tuple(origin) for origin in variant['origins']]
        yield (f"Recipe {{ opcode: {variant['opcode']}, count: {variant['count']}, "
               f"origins: &{origins}, checks: &{checks}, "
               f"sizes: &{variant['sizes']} }},")


def rider_source(path, variants):
    return '\n'.join(['use super::Recipe;', f'pub(super) const PATH: &str = "{path}";',
                      'pub(super) const RECIPES: &[Recipe] = &[',
                      *('    ' + entry for entry in rider_entries(variants)), '];', ''])


def dispatcher(name, slots, chunk, opcodes, variants):
    wide = name.startswith('t1536')
    cases = [f'case {op}: dispatch_body_{op}(addrs); break;' for op in opcodes]
    helpers = []
    if chunk == 0:
        cases.extend(f'case {op}: dispatch_body_1(addrs); break;' for op in range(41, 45))
        helpers.extend(rss_loop(v) for v in variants)
        cases.extend(f"case {v['opcode']}: loop_{v['opcode']}(addrs); break;" for v in variants)
    else:
        helpers.append(full_sequence('full_sequence', 4, wide))
        cases.extend(f'case {30+n}: full_sequence(addrs, {n}); break;' for n in range(2, 11))
        helpers.append(full_sequence('boundary_full_sequence', 70, wide))
        cases.extend(f'case {40+n}: boundary_full_sequence(addrs, {n}); break;' for n in range(1, 5))
    declarations = [f'void dispatch_body_{op}(uint64_t *addrs);' for op in opcodes]
    return '\n'.join(['#include <pe.h>', '#include <tuc.h>', *declarations, *helpers,
                      'void task_0(uint64_t *addrs, uint32_t _, uint16_t _profile_base_uid) {',
                      f'    switch ((addrs[{slots + 3}] & 0x1ff) >> 3) {{',
                      *('        ' + case for case in cases),
                      '        default: break;', '    }', '    trampoline(TRAMPOLINE_EXIT, 0, 0);', '}', ''])
