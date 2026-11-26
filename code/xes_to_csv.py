"""Stream BPI_Challenge_2019.xes.gz into a flat event table.

A full DOM parse needs more RAM than this machine has, so the log is read with
iterparse: pass 1 discovers the attribute schema from the first 3,000 traces,
pass 2 writes one CSV row per event. Trace-level attributes are prefixed
`case:`; event-level attributes keep their XES key.
"""
import csv, gzip, sys, time
import xml.etree.ElementTree as ET

SRC = sys.argv[1]
OUT = sys.argv[2]
ATTR = {"string", "date", "int", "float", "boolean", "id"}


def strip(t):
    return t.rsplit("}", 1)[-1]


def traces(path):
    """Yield (trace_attr_dict, [event_attr_dict, ...]) without holding the log."""
    with gzip.open(path, "rb") as fh:
        for ev, el in ET.iterparse(fh, events=("end",)):
            if strip(el.tag) != "trace":
                continue
            tattr, evs = {}, []
            for child in el:
                tag = strip(child.tag)
                if tag in ATTR:
                    tattr[child.get("key")] = child.get("value")
                elif tag == "event":
                    e = {}
                    for a in child:
                        if strip(a.tag) in ATTR:
                            e[a.get("key")] = a.get("value")
                    evs.append(e)
            yield tattr, evs
            el.clear()


t0 = time.time()
tkeys, ekeys = {}, {}
for i, (ta, evs) in enumerate(traces(SRC)):
    for k in ta:
        tkeys[k] = None
    for e in evs:
        for k in e:
            ekeys[k] = None
    if i >= 3000:
        break
cols = ["case:" + k for k in tkeys] + list(ekeys)
print(f"schema: {len(tkeys)} trace attrs, {len(ekeys)} event attrs ({time.time()-t0:.0f}s)", flush=True)

n_tr = n_ev = 0
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for ta, evs in traces(SRC):
        base = {"case:" + k: v for k, v in ta.items()}
        n_tr += 1
        for e in evs:
            row = dict(base)
            row.update(e)
            w.writerow(row)
            n_ev += 1
        if n_tr % 50000 == 0:
            print(f"  {n_tr:,} traces / {n_ev:,} events  {time.time()-t0:.0f}s", flush=True)
print(f"DONE traces={n_tr:,} events={n_ev:,} cols={len(cols)} {time.time()-t0:.0f}s", flush=True)
