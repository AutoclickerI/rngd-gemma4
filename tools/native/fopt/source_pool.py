"""Pool equal generated readonly literals before GCC generates address arithmetic.

Unlike post-link data movement, all uses of the canonical definitions are
compiled again. No function body or command byte is rewritten. Distinct pointer
identity must not be semantically significant for these generated constants.
"""
from collections import defaultdict
import hashlib
import re

ARRAY=re.compile(r'alignas\((\d+)\) const static uint8_t (\w+)\[\] = \{(.*?)\};',re.S)


def emitted_objects(elf):
    from elftools.elf.elffile import ELFFile
    result=set();owner=None
    with open(elf,'rb') as f:
        for symbol in ELFFile(f).get_section_by_name('.symtab').iter_symbols():
            if symbol['st_info']['type']=='STT_FILE':owner=symbol.name
            if symbol['st_info']['type']=='STT_OBJECT' and owner:
                unit=re.fullmatch(r'chunk(?:\d+)?_(\d+)\.c',owner)
                if unit:result.add((int(unit[1]),symbol.name))
    return result


def pool_sources(sources, live):
    groups=defaultdict(list)
    for unit,text in enumerate(sources):
        for m in ARRAY.finditer(text):
            if (unit,m[2]) not in live:continue
            if re.sub(r'0x[0-9a-fA-F]{2}|[\s,]','',m[3]):raise ValueError('nonliteral generated constant')
            data=bytes(int(x,16)for x in re.findall(r'0x([0-9a-fA-F]{2})',m[3]))
            groups[(int(m[1]),data)].append((unit,m))
    replacements=defaultdict(list);definitions=[]
    for (alignment,data),uses in sorted(groups.items(),key=lambda x:(hashlib.sha256(x[0][1]).hexdigest(),x[0][0])):
        if len(uses)<2:continue
        digest=hashlib.sha256(data).hexdigest();symbol='codesize_pool_'+digest+'_a'+str(alignment)
        attr=f'__attribute__((aligned({alignment}), visibility("hidden")))'
        definitions.append(f'const uint8_t {symbol}[{len(data)}] {attr} = {{'+','.join(f'0x{x:02x}'for x in data)+'};\n')
        for unit,m in uses:
            declaration=f'extern const uint8_t {symbol}[{len(data)}] {attr};\n#define {m[2]} {symbol}'
            replacements[unit].append((m.start(),m.end(),declaration))
    result=[]
    for unit,text in enumerate(sources):
        for start,end,replacement in sorted(replacements[unit],reverse=True):text=text[:start]+replacement+text[end:]
        result.append(text)
    if definitions:result.append('#include <stdint.h>\n'+''.join(definitions))
    return result
