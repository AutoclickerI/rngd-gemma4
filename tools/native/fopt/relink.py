#!/usr/bin/env python3
"""Reproduce an FOPT task blob from the PE program's relocatable ELF (gcc -r output + libpe).
Layout: alloc sections except .eh_frame, in section order, each at its alignment, base 0; RELA applied.
usage: relink.py <output.elf> [task.bin to compare] [--out blob.bin]"""
import io, struct, sys
from pathlib import Path
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection

ENTRY = 'task_0'

def _signed(value, bits, what):
    if not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
        raise ValueError(f'{what}: relocation overflow ({value}, signed {bits} bits)')
    return value


def _branch(value, bits, what):
    if value % 4:
        raise ValueError(f'{what}: unaligned branch displacement {value}')
    return (_signed(value, bits + 2, what) >> 2) & ((1 << bits) - 1)


def link(path, base=0):
    if not isinstance(base, int) or not 0 <= base < 2**64 or base % 4:
        raise ValueError('base must be an aligned unsigned 64-bit address')
    f = ELFFile(io.BytesIO(Path(path).read_bytes()))
    if (f.elfclass != 64 or not f.little_endian or
            f['e_machine'] != 'EM_AARCH64' or f['e_type'] != 'ET_REL'):
        raise ValueError('expected a little-endian AArch64 ELF64 relocatable object')
    secs = [s for s in f.iter_sections() if s['sh_flags'] & 2 and s.name != '.eh_frame' and s['sh_type'] in ('SHT_PROGBITS', 'SHT_NOBITS')]
    addr, at = {}, 4   # offset 0 holds the driver's `b <entry>` stub
    for s in secs:
        if s.name in addr:
            raise ValueError(f'duplicate allocated section name {s.name}')
        # The current task envelope has no independent BSS allocation/zeroing
        # contract. Omitting its bytes aliases later sections; do not pack it.
        if s['sh_type'] == 'SHT_NOBITS' and s['sh_size']:
            raise ValueError(f'nonempty NOBITS section {s.name} is unsupported')
        al = max(s['sh_addralign'], 1)
        if s.name != '.text':
            at = (at + al - 1) // al * al   # the driver's .text starts immediately after the branch stub
        addr[s.name] = at
        at += s['sh_size'] if s['sh_type'] == 'SHT_PROGBITS' else 0
    if base + at > 2**64:
        raise ValueError('task allocation overflows the address space')
    out = bytearray(at)
    for s in secs:
        if s['sh_type'] == 'SHT_PROGBITS': out[addr[s.name]:addr[s.name] + s['sh_size']] = s.data()
    symtab = f.get_section_by_name('.symtab'); undef = set(); kinds = {}
    if symtab is None:
        raise ValueError('missing symbol table')
    ent = [s for s in symtab.iter_symbols() if s.name == ENTRY]
    if len(ent) != 1 or not isinstance(ent[0]['st_shndx'], int):
        raise ValueError(f'expected one defined entry symbol {ENTRY}')
    entry_section = f.get_section(ent[0]['st_shndx'])
    if (entry_section.name not in addr or not entry_section['sh_flags'] & 4 or
            not 0 <= ent[0]['st_value'] <= entry_section['sh_size'] - 4):
        raise ValueError(f'entry symbol {ENTRY} is not inside executable task storage')
    e = addr[entry_section.name] + ent[0]['st_value']
    struct.pack_into('<I', out, 0, 0x14000000 | _branch(e, 26, 'entry stub'))
    for rs in f.iter_sections():
        if not isinstance(rs, RelocationSection): continue
        tgt = f.get_section(rs['sh_info'])
        if tgt.name not in addr: continue
        if not rs.is_RELA():
            raise ValueError(f'expected explicit RELA addends in {rs.name}')
        relocation_symbols = f.get_section(rs['sh_link'])
        for r in rs.iter_relocations():
            sym = relocation_symbols.get_symbol(r['r_info_sym']); t = r['r_info_type']; kinds[t] = kinds.get(t, 0) + 1
            what = f'reloc {t} in {tgt.name}+{r["r_offset"]:#x}'
            if t not in (257, 258, 261, 275, 277, 278, 279, 280, 282, 283, 284, 285, 286, 299):
                raise ValueError(f'{what}: unhandled relocation type')
            width = 8 if t == 257 else 4
            if not 0 <= r['r_offset'] <= tgt['sh_size'] - width:
                raise ValueError(f'{what}: relocation write exceeds target section')
            if sym['st_shndx'] == 'SHN_UNDEF': undef.add(sym.name); continue
            if not isinstance(sym['st_shndx'], int):
                raise ValueError(f'{what}: unsupported symbol section {sym["st_shndx"]}')
            sec = f.get_section(sym['st_shndx']).name
            if sec not in addr: raise ValueError(f'reloc into dropped section {sec}')
            S = base + addr[sec] + sym['st_value']; A = r['r_addend']; P = base + addr[tgt.name] + r['r_offset']
            o = addr[tgt.name] + r['r_offset']; ins, = struct.unpack_from('<I', out, o)
            if t == 257:   struct.pack_into('<Q', out, o, (S + A) & (2**64 - 1))           # ABS64
            elif t == 258:                                                               # ABS32
                if not -(2**31) <= S + A < 2**32:
                    raise ValueError(f'{what}: relocation overflow ({S + A}, ABS32)')
                struct.pack_into('<I', out, o, (S + A) & 0xffffffff)
            elif t == 261: struct.pack_into('<i', out, o, _signed(S + A - P, 32, what))    # PREL32
            elif t == 275:                                                                 # ADR_PREL_PG_HI21
                v = _signed(((S + A) >> 12) - (P >> 12), 21, what) & 0x1fffff
                struct.pack_into('<I', out, o, (ins & 0x9f00001f) | ((v & 3) << 29) | ((v >> 2) << 5))
            elif t in (277, 278, 284, 285, 286, 299):                                      # ADD/LDST{8,16,32,64,128}_ABS_LO12_NC
                sh = {277: 0, 278: 0, 284: 1, 285: 2, 286: 3, 299: 4}[t]
                if (S + A) % (1 << sh):
                    raise ValueError(f'{what}: unaligned load/store address {S + A}')
                v = ((S + A) & 0xfff) >> sh
                struct.pack_into('<I', out, o, (ins & ~(0xfff << 10) & 0xffffffff) | (v << 10))
            elif t in (282, 283):                                                          # JUMP26 / CALL26
                v = _branch(S + A - P, 26, what)
                struct.pack_into('<I', out, o, (ins & 0xfc000000) | v)
            elif t in (280,):                                                              # CONDBR19
                v = _branch(S + A - P, 19, what)
                struct.pack_into('<I', out, o, (ins & 0xff00001f) | (v << 5))
            elif t in (279,):                                                              # TSTBR14
                v = _branch(S + A - P, 14, what)
                struct.pack_into('<I', out, o, (ins & 0xfff8001f) | (v << 5))
    return bytes(out), addr, undef, kinds

if __name__ == '__main__':
    blob, addr, undef, kinds = link(sys.argv[1])
    print('layout', {k: hex(v) for k, v in addr.items()}, 'size', len(blob), 'undefined', sorted(undef), 'reloc kinds', kinds)
    if undef:
        raise SystemExit(f'unresolved symbols: {", ".join(sorted(undef))}')
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        ref = open(sys.argv[2], 'rb').read()
        n = min(len(ref), len(blob)); d = [i for i in range(n) if ref[i] != blob[i]]
        print(f'ref {len(ref)} B, ours {len(blob)} B, differing bytes in common range: {len(d)}', d[:12], 'tail of ref beyond ours nonzero:', any(ref[len(blob):]))
    if '--out' in sys.argv: open(sys.argv[sys.argv.index('--out') + 1], 'wb').write(blob)
