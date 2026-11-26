#!/usr/bin/env python3
"""Issue intra-cluster PE all-to-all DMA in permutation rounds.

Opt groups the four destination PEs by round, creating an incast at each
destination. In a permutation round the engines write to distinct destination
PEs. Each engine still issues the same descriptors and the same waits.

What changes: only the order in which each engine takes its four descriptors. Every (descriptor, source, destination)
triple stays on the engine (slot) that issued it, run tags and operands stay in place, so the set of transfers and the
completion ids are those of the stock code. An exchange is one dedup call of four runs or, in a recompiled body (call
dedup / queue-block packing), four runs issued across consecutive call sites and inline run_dma_static commands. The
tool refuses unless
  (a) every rotated exchange is a full 4 x 4 all-to-all whose every run is an incast;
  (b) the first wait group after the exchange on any of its run tags (consecutive wait_dma commands, dedup calls
      expanded, with only profile markers / queue-block bookkeeping between them) waits on all of them, so no consumer
      depends on which run carried which transfer;
  (c) a split exchange holds, from its first descriptor copy to its last run, only descriptor alloc/copy/publish, its
      own runs and queue-block bookkeeping (no other TUC command, TIDX_DYNAMIC_SLOT assignment or code section); each
      descriptor slot is its source PE; each rewritten call argument is rewritten once and feeds exactly one use in its
      function body;
  (d) the unmodified task C rebuilds the stock image byte for byte, and the rewrite keeps the task C's line count
      (CHECK() __LINE__ immediates) and the image length;
  (e) self-check: the rewritten task C holds the same exchanges with the same 16 triples each, the rotated ones now in
      permutation rounds (each run reaches four distinct PEs).

"""
from itertools import permutations
import re
import sys

from direct_build import build

CALL = re.compile(r'tail = (function_chip\d+_cluster\d+_\d+)\(tail,')
DEF = re.compile(r'^static uint64_t (function_chip\d+_cluster\d+_\d+)\(uint64_t tail(.*)\) \{$', re.M)
DMA = re.compile(r'\s*dma_\d+\s*')


def split_args(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '({[':
            depth += 1
        elif ch in ')}]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur); cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def close(src, i):
    """Index just past the parenthesis closing the call whose argument list starts at i."""
    depth = 1
    while depth:
        if src[i] in '({[':
            depth += 1
        elif src[i] in ')}]':
            depth -= 1
        i += 1
    return i


def pe(a):
    return int(re.search(r'CHIP_CLUSTER_PE\(\d+, \d+, (\d)\)', a).group(1))


def remote(t, own):
    for a in (t[1], t[2]):
        m = re.search(r'LOCAL_CHIP_SAME_CLUSTER\((\d)\)', a)
        if m:
            return int(m.group(1))
    return own


def scan(src):
    """All-to-all calls: (start, end, cluster, runs, slots, remotes, tags, incast)."""
    found = []
    for m in CALL.finditer(src):
        i = m.end(); j = close(src, i); args = split_args(src[i:j - 1])
        if not any(DMA.fullmatch(a) for a in args):
            continue
        runs, k, ok = [], 0, True
        while k < len(args):
            trip = []
            while k < len(args) and DMA.fullmatch(args[k]):
                if k + 2 >= len(args) or 'get_sram_address' not in args[k + 1] or 'get_sram_address' not in args[k + 2]:
                    ok = False; break
                trip.append((args[k].strip(), args[k + 1].strip(), args[k + 2].strip())); k += 3
            if not ok or len(trip) != 4 or k + 1 >= len(args) or 'dma_w_operand0_t' not in args[k]:
                ok = False; break
            runs.append((trip, args[k].strip(), args[k + 1].strip())); k += 2
        if not ok or len(runs) != 4:
            continue
        slots = [pe(t[1]) for t in runs[0][0]]
        if sorted(slots) != [0, 1, 2, 3] or any([pe(t[1]) for t in r[0]] != slots for r in runs):
            continue
        rem = [[remote(t, slots[s]) for s, t in enumerate(r[0])] for r in runs]
        if not all(sorted(rem[g][s] for g in range(4)) == [0, 1, 2, 3] for s in range(4)):
            continue
        incast = all(len(set(rem[g])) == 1 for g in range(4))
        cluster = int(re.search(r'_cluster(\d+)_', m.group(1)).group(1))
        tags = [int(r[2]) for r in runs]
        found.append((m.start(), j - 1, i, cluster, runs, slots, rem, tags, incast))
    return found


BODYPARAMS = {}


def bodytext(src):
    """dedup function -> (parameter names after tail, body text)."""
    out = {}
    for m in DEF.finditer(src):
        params = [p.strip().split()[-1].lstrip('*') for p in m.group(2).split(',')[1:]] if m.group(2) else []
        out[m.group(1)] = (params, src[m.end():src.index('\n}\n', m.end())])
    return out


def selfcheck(src, out, n):
    """(e) every exchange of the output is the input's with the same 16 (descriptor, source, destination) triples, each
    rotated one now in permutation rounds (each run reaches four distinct PEs) and the others unchanged."""
    def ex(s):
        res = []
        for _, _, runs in exchanges(s, stream(s)):
            trip = [[triple(c) for c in r[0]] for r in runs]
            if any(DMA.fullmatch(' ' + t[0] + ' ') is None or 'get_sram_address' not in t[1] for r in trip for t in r):
                continue
            own = [pe(t[1]) for t in trip[0]]
            res.append((sorted(t for r in trip for t in r), [len({remote(t, own[k]) for k, t in enumerate(r)}) for r in trip]))
        return res
    a, b = ex(src), ex(out)
    if len(a) != len(b) or any(x[0] != y[0] for x, y in zip(a, b)):
        sys.exit('self-check: the rewrite changed the set of exchange transfers')
    rot = [(x, y) for x, y in zip(a, b) if x[1] != y[1]]
    if len(rot) != n or any(y[1] != [4, 4, 4, 4] for _, y in rot):
        sys.exit(f'self-check: {len(rot)} exchanges changed, {n} expected, or not all in permutation rounds')


TASK = re.compile(r'^static void (task_\w+)\(', re.M)
STMT = re.compile(r'tail = (\w+)\(tail|tuc_sync_tail\(tail\)|tail = tuc_reserve_resources\(|TIDX_DYNAMIC_SLOT\(\w+\) =|'
                  r'// \d+ \(code\)|(?<![\w.>])(alloc_dma_descriptor|publish_arena_dma_descriptor)\(\)|'
                  r'(?<![\w.>])(copy_dma_descriptor|push_core_interrupt_command|wait_sync_intra_chip_cluster|'
                  r'sync_intra_chip_cluster|set_indirect_list_address|record_dma_push|wait_core_interrupt_command|'
                  r'tuc_wait|stage_pdma_desc|wait_pdma_desc)\(|(?<![\w.>])(enter_profile|leave_profile)\(')
BODY = re.compile(r'tail = wait_dma\(tail, (\w+)\)|tail = run_dma_static\(tail,|copy_dma_descriptor\(&shared->dma_desc\[(\d)\], '
                  r'(\w+), 0, (\w+), (\w+)\)|(alloc_dma_descriptor|publish_arena_dma_descriptor)\(\)|tail = (\w+)\(tail|'
                  r'(?<![\w.>])(push_core_interrupt_command|wait_sync_intra_chip_cluster|sync_intra_chip_cluster|'
                  r'set_indirect_list_address|record_dma_push|copy_dma_descriptor)\(|(enter_profile|leave_profile)\(')


def bodies(src):
    """dedup function -> (param names, [events]); events: ('W', param) wait, ('R', operand param|None, tag param|None) run,
    ('C', slot, (buf, src, dst) params) descriptor copy, ('A',) alloc/publish, ('P',) profile, ('O', what) other command."""
    out = {}
    for m in DEF.finditer(src):
        params = [p.strip().split()[-1].lstrip('*') for p in m.group(2).split(',')[1:]] if m.group(2) else []
        b0 = m.end(); body = src[b0:src.index('\n}\n', b0)]; ev = []
        for e in BODY.finditer(body):
            if e.group(1):
                ev.append(('W', e.group(1)))
            elif e.group(0).startswith('tail = run_dma_static'):
                i = e.end(); args = split_args(body[i:close(body, i) - 1])
                ev.append(('R', args[-2].strip(), args[-1].strip()))
            elif e.group(2):
                ev.append(('C', int(e.group(2)), (e.group(3), e.group(4), e.group(5))))
            elif e.group(6):
                ev.append(('A',))
            elif e.group(9):
                ev.append(('P',))
            else:
                ev.append(('O', e.group(7) or e.group(8)))
        out[m.group(1)] = (params, ev)
    return out


def stream(src):
    """The TUC command stream of every task body, dedup calls expanded: list of (pos, kind, data)."""
    fn = bodies(src); ev = []
    for t in TASK.finditer(src):
        b0 = t.end(); b1 = src.index('\n}\n', b0)
        for m in STMT.finditer(src, b0, b1):
            g = m.group(0); pos = m.start()
            if m.group(1) and m.group(1).startswith('function_chip'):
                i = m.end() + 1; j = close(src, i); args = split_args(src[i:j - 1])
                params, body = fn[m.group(1)]; idx = {p: k for k, p in enumerate(params)}
                for e in body:
                    if e[0] == 'W':
                        v = args[idx[e[1]]].strip() if e[1] in idx else e[1]
                        ev.append((pos, 'W', int(v) if v.isdigit() else v))
                    elif e[0] == 'R':
                        tag = args[idx[e[2]]].strip() if e[2] in idx else e[2]
                        ev.append((pos, 'R', tag))
                    elif e[0] == 'C':
                        # A dedup body may retain its immutable DMA descriptor symbol
                        # while parameterizing both addresses. Only that known literal
                        # may bypass call arguments; addresses must still be parameters.
                        refs = [idx[p] if p in idx else p for p in e[2]]
                        if not (isinstance(refs[0], int) or DMA.fullmatch(refs[0])) or not all(isinstance(v, int) for v in refs[1:]):
                            ev.append((pos, 'Cfix', e[1])); continue
                        ev.append((pos, 'C', (e[1], m.group(1), i, j - 1, refs, args)))
                    elif e[0] in 'AP':
                        ev.append((pos, e[0], None))
                    else:
                        ev.append((pos, 'O', e[1]))
            elif m.group(1) == 'wait_dma':
                i = m.end() + 1; v = split_args(src[i:close(src, i) - 1])[0].strip()
                ev.append((pos, 'W', int(v) if v.isdigit() else v))
            elif m.group(1) == 'run_dma_static':
                i = m.end() + 1; args = split_args(src[i:close(src, i) - 1])
                ev.append((pos, 'R', args[-1].strip()))
            elif g.startswith('tuc_sync_tail') or g.startswith('tail = tuc_reserve_resources'):
                ev.append((pos, 'Q', None))
            elif g.startswith('TIDX_DYNAMIC_SLOT') or g.startswith('//'):
                ev.append((pos, 'X', g))
            elif m.group(2):
                ev.append((pos, 'A', None))
            elif m.group(4):
                ev.append((pos, 'P', None))
            elif m.group(3) == 'copy_dma_descriptor':
                ev.append((pos, 'Cfix', None))
            else:
                ev.append((pos, 'O', m.group(1) or m.group(3)))
    return ev


def wait_group(ev, k, tags):
    """Union of the first wait group after stream index k that touches `tags`: consecutive waits with no TUC command
    between them (profile markers and queue-block bookkeeping -- tuc_sync_tail / tuc_reserve_resources, which queue
    commands but are none -- may sit between; T0's {2, 3} | block boundary | {4, 5} is one group)."""
    tags = set(tags)
    for n in range(k + 1, len(ev)):
        if ev[n][1] == 'W' and ev[n][2] in tags:
            got = set(); m = n
            while m < len(ev) and ev[m][1] in ('W', 'P', 'Q'):
                if ev[m][1] == 'W':
                    got.add(ev[m][2])
                m += 1
            return got
    return None


def exchanges(src, ev):
    """Runs gathered across call sites: [(first C index, last R index, runs)] with runs = [[C events in slot order], tag]."""
    found = []; n = 0
    while n < len(ev):
        if ev[n][1] != 'C':
            n += 1; continue
        runs, cur, m, ok = [], [], n, True
        while m < len(ev) and len(runs) < 4:
            k = ev[m][1]
            if k == 'C':
                cur.append(ev[m])
            elif k == 'R':
                if len(cur) != 4:
                    ok = False; break
                runs.append((cur, ev[m][2])); cur = []
            elif k not in 'AQP':
                ok = False; break
            m += 1
        if ok and len(runs) == 4 and not cur:
            found.append((n, m - 1, runs)); n = m
        else:
            n += 1
    return found


def triple(c):
    _, _, _, _, ix, args = c[2]
    return tuple(args[x].strip() if isinstance(x, int) else x for x in ix)


def rotate(src, order):
    BODYPARAMS.clear(); BODYPARAMS.update(bodytext(src))
    calls = scan(src); ev = stream(src); report = []
    stock_spans = {(c[2], c[1]) for c in calls}
    edits = {}    # (argstart, end) of a call site -> {arg index: new text}
    stock_calls = []
    for start, end, argstart, cluster, runs, slots, rem, tags, incast in calls:
        if not incast:
            report.append(f'cluster {cluster}: all-to-all at line {src.count(chr(10), 0, start) + 1} not incast: kept'); continue
        last = max(n for n, e in enumerate(ev) if e[0] == start)
        w = wait_group(ev, last, tags)
        if w is None or not set(tags) <= w:
            sys.exit(f'line {src.count(chr(10), 0, start) + 1}: the first wait group on run tags {tags} waits {w}: refusing')
        stock_calls.append((start, end, argstart, cluster, runs, slots, rem, tags, incast))
    for n0, n1, runs in exchanges(src, ev):
        sites = sorted({(c[2][2], c[2][3]) for r in runs for c in r[0]})
        if len(sites) == 1 and sites[0] in stock_spans:
            continue                                   # whole-call exchange: the stock code path above
        trip = [[triple(c) for c in r[0]] for r in runs]
        if any(DMA.fullmatch(' ' + t[0] + ' ') is None or 'get_sram_address' not in t[1] or 'get_sram_address' not in t[2]
               for r in trip for t in r):
            continue
        slots = [pe(t[1]) for t in trip[0]]
        if sorted(slots) != [0, 1, 2, 3] or any([pe(t[1]) for t in r] != slots for r in trip):
            continue
        if any(c[2][0] != pe(triple(c)[1]) for r in runs for c in r[0]):
            continue                                   # descriptor slot must be the source PE (the engine)
        rem = [[remote(t, slots[s]) for s, t in enumerate(r)] for r in trip]
        if not all(sorted(rem[g][s] for g in range(4)) == [0, 1, 2, 3] for s in range(4)):
            continue
        cl = int(re.search(r'_cluster(\d+)_', runs[0][0][0][2][1]).group(1)); line = src.count('\n', 0, ev[n0][0]) + 1
        if not all(len(set(rem[g])) == 1 for g in range(4)):
            report.append(f'cluster {cl}: split all-to-all at line {line} not incast: kept'); continue
        tags = [int(t) for _, t in runs]
        if any(e[1] == 'X' for e in ev[n0:n1 + 1]):
            sys.exit(f'line {line}: a code section / TIDX assignment inside the split exchange: refusing')
        w = wait_group(ev, n1, tags)
        if w is None or not set(tags) <= w:
            sys.exit(f'line {line}: the first wait group on split run tags {tags} waits {w}: refusing')
        for j in range(4):
            for s in range(4):
                g = [gg for gg in range(4) if rem[gg][s] == (slots[s] + order[j]) % 4]
                assert len(g) == 1
                c = runs[j][0][s]; _, fname, a0, a1, ix, args = c[2]
                for p in (BODYPARAMS[fname][0][x] for x in ix if isinstance(x, int)):
                    if len(re.findall(r'\b' + p + r'\b', BODYPARAMS[fname][1])) != 1:
                        sys.exit(f'line {line}: parameter {p} of {fname} is used more than once in its body: refusing')
                for x, new in zip(ix, trip[g[0]][s]):
                    if not isinstance(x, int):
                        if x != new:
                            sys.exit(f'line {line}: rotation would change fixed descriptor {x} to {new}: refusing')
                        continue
                    if x in edits.get((a0, a1), {}):
                        sys.exit(f'line {line}: argument {x} of the call at {a0} would be rewritten twice: refusing')
                    old = args[x]; edits.setdefault((a0, a1), {})[x] = old[:len(old) - len(old.lstrip())] + new + \
                        old[len(old.rstrip()):]
        names = sorted({t[0] for r in trip for t in r}); nsite = len(sites)
        report.append(f'cluster {cl}: line {line} {",".join(names)} tags {tags} (split over {nsite} call sites): '
                      f'incast rounds -> permutation rounds')
    # Preserve the whole-call formatting used by the qualified task C.
    pieces, pos = [], 0
    allspans = sorted([(c[2], c[1], 'stock', c) for c in stock_calls] + [(a0, a1, 'split', e) for (a0, a1), e in edits.items()])
    for a0, a1, kind, data in allspans:
        if a0 < pos:
            sys.exit('overlapping rewrites')
        if kind == 'stock':
            start, end, argstart, cluster, runs, slots, rem, tags, incast = data; new = []
            for jrun in range(4):
                for s in range(4):
                    g = [gg for gg in range(4) if rem[gg][s] == (slots[s] + order[jrun]) % 4]
                    new += list(runs[g[0]][0][s])
                new += [runs[jrun][1], runs[jrun][2]]
            pieces += [src[pos:argstart], ' ' + ', '.join(new)]
            report.append(f'cluster {cluster}: line {src.count(chr(10), 0, start) + 1} '
                          f'{",".join(sorted({t[0] for r in runs for t in r[0]}))} tags {tags}: incast rounds -> permutation rounds')
        else:
            args = split_args(src[a0:a1])
            pieces += [src[pos:a0], ','.join(data.get(x, a) for x, a in enumerate(args))]
        pos = a1
    pieces.append(src[pos:]); out = ''.join(pieces)
    if out.count('\n') != src.count('\n'):
        sys.exit('the rewrite changed the line count')
    return out, report, len(stock_calls) + sum(1 for r in report if 'split over' in r)


def rebuild_rotated(base, task, output, task_out):
    """Use the first equivalent permutation that fits the stock image."""
    source = task.read_text()
    size = base.stat().st_size
    for order in permutations(range(4)):
        rotated, report, count = rotate(source, order)
        assert count, 'no incast exchange'
        selfcheck(source, rotated, count)
        task_out.write_text(rotated)
        build(base, task_out, output)
        if output.stat().st_size == size:
            print(*report, sep='\n')
            print(f'{count} exchanges rotated with order {order}')
            return
    raise ValueError('all equivalent round orders changed image length')
