"""Lower only the packed 15-packet Gate/Up records to native INT4.

The fields come from probe pairs compiled both ways
(g1_i8_f2 / g1_i4_f2w and g1_ug8 / g1_ug4: an Up/Gate pass with the switch ring, the epilogue and
two-flit packets `[K / CH % 2, K % U]`), whose task C differs in exactly these bytes:

  main_tu (dot-product pass, 1320- or 352-byte record):
    0x00e fetch_unit.last_dim_info.bytes[6]       3 -> 1   (fetch element type, i8 -> i4)
    0x097 dot_product_engine.feed_data_type       1 -> 0
    0x0b0 dot_product_engine.reg_indexer_strides  x2 (u32[8]: TRF strides in elements)
    0x0e9 dot_product_engine.mac_tree_depth       6 -> 7   (log2 of 128 elements per packet)
    0x0ea dot_product_engine.mac_tree_types       1 -> 0
    0x0f0 dot_product_engine.reg_types            2 -> 0
    0x0f1 dot_product_engine.reg_read_log_size    6 -> 7
  sub_tu (TRF store of the weight, 64-byte slot):
    0x008 fetch element type                      3 -> 1
"""
import re
import struct

ARRAY = re.compile(r'const static uint8_t (\w+)\[\] = \{\n(.*?)\n\};', re.S)
BYTE = re.compile(r'0x([0-9a-fA-F]{2})')
PACKED = 15


def arrays(source):
    return {name: bytes(int(value, 16) for value in BYTE.findall(body))
            for name, body in ARRAY.findall(source)}


def is_dot_i8(r):
    return len(r) > 0x0f2 and r[0x097] == 1 and r[0x0ea] == 1


def main_packets(r):
    return struct.unpack_from('<8H', r, 0x0a0), r[0x0e0], r[0x090]


def sub_limits(r):
    return struct.unpack_from('<8H', r, 0x010)


def patch_main(r):
    b = bytearray(r)
    assert b[0x090] == 2, f'flits_per_input {b[0x090]} (expected the two-flit i8 packet)'
    assert (b[0x00e], b[0x0e9], b[0x0ea], b[0x0f0], b[0x0f1]) == (3, 6, 1, 2, 6), \
        f'unrecognized i8 record: ldi6 {b[0x00e]} depth {b[0x0e9]} mtt {b[0x0ea]} regtypes {b[0x0f0]} rlog {b[0x0f1]}'
    b[0x00e] = 1
    b[0x097] = 0
    for i in range(8):
        v, = struct.unpack_from('<I', b, 0x0b0 + 4 * i)
        struct.pack_into('<I', b, 0x0b0 + 4 * i, 2 * v)
    b[0x0e9], b[0x0ea], b[0x0f0], b[0x0f1] = 7, 0, 0, 7
    return bytes(b)


def patch_sub(r):
    b = bytearray(r)
    b[0x008] = 1
    return bytes(b)


def lower(image, source, geometry=None):
    """Patch the same descriptors in the stock image and task C."""
    records = arrays(source)
    changes = {}
    main_count = sub_count = 0
    for name, record in sorted(records.items()):
        if name.startswith('main_tu_') and is_dot_i8(record):
            seq, acc, flits = main_packets(record)
            if seq[2] != PACKED and acc != 2 * PACKED:
                continue
            assert (seq[2], acc, flits) == (PACKED, 2 * PACKED, 2), name
            assert geometry is None or seq == geometry[0], name
            changes[name] = patch_main(record)
            main_count += 1
        elif name.startswith('sub_tu_') and len(record) == 64 and record[8] == 3:
            limits = sub_limits(record)
            if limits[2] != PACKED:
                continue
            assert geometry is None or limits == geometry[1], name
            changes[name] = patch_sub(record)
            sub_count += 1
    assert main_count and sub_count, 'missing packed GU operands'

    image = bytearray(image)
    seen = set()
    for name, new in changes.items():
        old = records[name]
        if old in seen:
            continue
        seen.add(old)
        offsets = []
        at = 0
        while (at := image.find(old, at)) >= 0:
            offsets.append(at)
            at += 1
        assert offsets, f'{name} not found in the image'
        for at in offsets:
            image[at:at + len(old)] = new

    def replace(match):
        name, body = match.groups()
        if name not in changes:
            return match[0]
        values = iter(changes[name])
        body = BYTE.sub(lambda _: f'0x{next(values):02x}', body)
        return f'const static uint8_t {name}[] = {{\n{body}\n}};'

    patched = ARRAY.sub(replace, source)
    assert arrays(patched) == records | changes
    print(f'lowered {main_count} Main contractions and {sub_count} Sub stores')
    return image, patched
