"""Read the Opt kernel registry emitted in the server ELF."""
import struct

SECTION = b"furiosa_kernels"

def section(elf):
    assert elf[:5] == b"\x7fELF\x02", "not a 64-bit ELF"
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3a)
    sh = lambda i: struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize)
    stroff = sh(shstrndx)[4]
    for i in range(shnum):
        s = sh(i); nm = elf[stroff + s[0]: elf.index(b"\0", stroff + s[0])]
        if nm == SECTION: return s[4], s[5]
    raise KeyError(SECTION.decode())

def decode(b):
    at = 0
    def q():
        nonlocal at; v, = struct.unpack_from("<Q", b, at); at += 8; return v
    rows = []
    for _ in range(q()):
        n = q(); path = b[at:at+n].decode(); at += n
        key = [q() for _ in range(q())]
        n = q(); img = bytes(b[at:at+n]); at += n
        rows.append((path, key, img))
    return rows, at
