"""Insert the narrow G12 members' weights and merge their image requirements."""

import dispatch_build as dispatch


def append_weights(paths, folder, foptool, member_file, offsets, base_weight, meta):
    weights = bytearray(base_weight)
    for op in (0, 2):
        parts = folder / f'member-{op}'
        parts.mkdir()
        member = dispatch.split(foptool, member_file(paths[op], 'bin'), parts)
        weight = member['weight_bytes']
        offset = next((i for i in range(0, len(weights), dispatch.WEIGHT_ALIGN)
                       if weights[i:i + len(weight)] == weight), None)
        if offset is None:
            weights.extend(bytes(-len(weights) % dispatch.WEIGHT_ALIGN))
            offset = len(weights)
            weights.extend(weight)
        offsets[op] = offset // dispatch.WEIGHT_ALIGN if weight else 0
        assert (meta['chips'], meta['pes']) == (member['chips'], member['pes'])
        for dest, src in zip(meta['stacks'], member['stacks'], strict=True):
            dest['size'] = max(dest['size'], src['size'])
        meta['sram_end'] = max(meta['sram_end'], member['sram_end'])
        for level, depth in member['profile']['depth'].items():
            meta['profile']['depth'][level] = max(meta['profile']['depth'][level], depth)
    meta['weight']['len'] = len(weights)
    return weights
