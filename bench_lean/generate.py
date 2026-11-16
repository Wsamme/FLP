"""Generate lean discrete-assembly plant instances.

Plant archetype (pump / motor / gearbox type products):
  receiving -> raw-material store / purchased-parts store
  raw material -> machining cells (housing, shaft, gear) -> shared heat treatment -> washing -> line-side supermarket
  raw material -> welding -> painting (frames, only some families)
  purchased parts -> supermarket ; supermarket -> sub-assembly lines and final lines (tugger milk run)
  sub-assembly -> final assembly -> end-of-line test -> packing -> finished-goods store -> shipping
  support: tool crib, maintenance, QC lab, office, locker/canteen (people flows)

Areas come from capacity (machines from load hours, line length from takt, stores from inventory days),
the from-to matrix comes from products x routings x volumes / unit loads, and every flow carries its
handling equipment. All parameters are in params.py with their source tag.

Usage: python -m bench_lean.generate   (writes bench_lean/instances/*.json)
"""
import json, math, os
import numpy as np
from . import params as PR

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances")

TIERS = {
    "S": dict(families=2, cells={"housing": 1, "shaft": 1}, wash=1, weld=0, paint=0, sub=2, final=1, sm=1,
              vol=(20000, 40000), maint=False),
    "M": dict(families=4, cells={"housing": 2, "shaft": 1, "gear": 1}, wash=1, weld=1, paint=1, sub=3, final=2, sm=2,
              vol=(10000, 30000), maint=True),
    "L": dict(families=6, cells={"housing": 3, "shaft": 2, "gear": 2}, wash=2, weld=2, paint=1, sub=5, final=3, sm=3,
              vol=(8000, 22000), maint=True),
}
# machined part types: operations in the cell (machine footprint m2, cycle min range), then shared steps
PARTS = {
    "housing": dict(name="壳体", ops=[("车", 8.75, (2, 5)), ("铣", 12.0, (3, 7)), ("钻", 7.5, (1, 3))], heat=False,
                    blank_per_pallet=(8, 16), parts_per_bin=(4, 8)),
    "shaft": dict(name="轴", ops=[("车", 8.75, (1.5, 4)), ("磨", 10.0, (2, 4))], heat=True,
                  blank_per_pallet=(40, 80), parts_per_bin=(20, 40)),
    "gear": dict(name="齿轮", ops=[("车", 8.75, (1.5, 3.5)), ("滚齿", 10.0, (3, 6)), ("磨", 10.0, (2, 4))], heat=True,
                 blank_per_pallet=(30, 60), parts_per_bin=(15, 30)),
}
HEAT_FOOT, HEAT_MIN = 40.0, (0.4, 0.8)        # ASM: furnace line footprint m2, min per part (batch equivalent)
WASH_FOOT, WASH_MIN = 18.0, (0.2, 0.4)
WELD_FOOT, WELD_MIN = 30.0, (5, 10)           # robot welding cell
PAINT_AREA = {"M": 360.0, "L": 540.0}          # ASM: conveyorised paint line incl. booths and oven
SUB_WORK, FINAL_WORK, TEST_MIN = (8, 18), (20, 40), (6, 12)


def U(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def build(tier, pattern, seed, brownfield=False):
    rng = np.random.default_rng(seed)
    T = TIERS[tier]
    avail_min = PR.SHIFTS * PR.HOURS * 60.0               # per day
    depts, D = [], {}

    def add(key, name, typ, area, **kw):
        d = dict(id=len(depts), key=key, name=name, type=typ, area=round(float(area), 1),
                 aspect_max=kw.pop("aspect_max", PR.ASPECT_DEFAULT), min_side=kw.pop("min_side", PR.MIN_SIDE),
                 headcount_shift=int(kw.pop("head", 0)), **kw)
        depts.append(d); D[key] = d["id"]; return d["id"]

    # ---- products ----
    fams = []
    for f in range(T["families"]):
        annual = int(round(U(rng, T["vol"]), -2))
        fams.append(dict(id=f, name=f"产品族 {chr(65 + f)}", annual_volume=annual, daily=annual / PR.DAYS,
                         frame=(T["weld"] > 0 and f % 2 == 0),
                         parts={pt: int(rng.integers(1, 3)) for pt in T["cells"]},
                         purchased_kits_per_unit=1, pp_parts_per_unit=int(rng.integers(12, 30))))
    # cell assignment: family f -> cell index f mod count
    flows = []          # (from_key, to_key, trips/day, equipment, what)

    def flow(a, b, trips, eq, what):
        if trips > 0: flows.append([a, b, float(trips), eq, what])

    # ---- machining cells ----
    cell_load = {}   # (part, idx) -> minutes per day per op
    unit = {}
    for pt in T["cells"]:
        unit[pt] = dict(pallet=int(rng.integers(*PARTS[pt]["blank_per_pallet"])), bin=int(rng.integers(*PARTS[pt]["parts_per_bin"])))
    cyc = {pt: [U(rng, o[2]) for o in PARTS[pt]["ops"]] for pt in T["cells"]}
    for f in fams:
        for pt, q in f["parts"].items():
            k = f["id"] % T["cells"][pt]
            cell_load.setdefault((pt, k), np.zeros(len(PARTS[pt]["ops"])))
            cell_load[(pt, k)] += np.array(cyc[pt]) * f["daily"] * q
    heat_min = U(rng, HEAT_MIN); wash_min = U(rng, WASH_MIN)
    heat_parts = wash_parts = 0.0
    for (pt, k), load in sorted(cell_load.items()):
        machines = [max(1, math.ceil(l / (avail_min * PR.OEE))) for l in load]
        foot = sum(m * o[1] for m, o in zip(machines, PARTS[pt]["ops"]))
        add(f"cell_{pt}_{k}", f"{PARTS[pt]['name']}加工单元 {k + 1}", "cell", foot * (1 + PR.MACHINE_ALLOW),
            head=math.ceil(sum(machines) * 0.5), machines=dict(zip([o[0] for o in PARTS[pt]["ops"]], machines)))
    for f in fams:
        for pt, q in f["parts"].items():
            n = f["daily"] * q
            wash_parts += n
            if PARTS[pt]["heat"]: heat_parts += n
    heat_m = max(1, math.ceil(heat_parts * heat_min / (avail_min * PR.OEE)))
    add("heat", "热处理", "monument", heat_m * HEAT_FOOT * (1 + PR.MACHINE_ALLOW), head=2 * heat_m, hazard="热源")
    for w in range(T["wash"]):
        wm = max(1, math.ceil(wash_parts / T["wash"] * wash_min / (avail_min * PR.OEE)))
        add(f"wash_{w}", f"清洗 {w + 1}", "monument", wm * WASH_FOOT * (1 + PR.MACHINE_ALLOW), head=1)
    # ---- welding / painting ----
    weld_min = U(rng, WELD_MIN); frames = sum(f["daily"] for f in fams if f["frame"])
    for w in range(T["weld"]):
        cells = max(1, math.ceil(frames / T["weld"] * weld_min / (avail_min * PR.OEE)))
        add(f"weld_{w}", f"焊接 {w + 1}", "cell", cells * WELD_FOOT * (1 + PR.MACHINE_ALLOW), head=cells, hazard="明火")
    if T["paint"]:
        add("paint", "喷漆", "monument", PAINT_AREA[tier], head=4, hazard="易燃涂料", exterior_wall=True, aspect_max=5.0)
    # ---- assembly lines (takt) ----
    for s in range(T["sub"]):
        served = [f for f in fams if f["id"] % T["sub"] == s or (f["id"] + 1) % T["sub"] == s]
        dem = sum(f["daily"] for f in served) or fams[0]["daily"]
        takt = avail_min / dem
        st = max(2, math.ceil(U(rng, SUB_WORK) / (takt * PR.LINE_EFF)))
        L = st * PR.STATION_PITCH
        add(f"sub_{s}", f"部装线 {s + 1}", "line", L * PR.LINE_WIDTH, head=st,
            aspect_max=max(PR.ASPECT_DEFAULT, round(L / PR.LINE_WIDTH + 0.5, 1)), min_side=PR.LINE_WIDTH, stations=st)
    for a in range(T["final"]):
        served = [f for f in fams if f["id"] % T["final"] == a]
        dem = sum(f["daily"] for f in served)
        takt = avail_min / dem
        st = max(4, math.ceil(U(rng, FINAL_WORK) / (takt * PR.LINE_EFF)))
        L = st * PR.STATION_PITCH
        add(f"final_{a}", f"总装线 {a + 1}", "line", L * PR.LINE_WIDTH, head=st,
            aspect_max=max(PR.ASPECT_DEFAULT, round(L / PR.LINE_WIDTH + 0.5, 1)), min_side=PR.LINE_WIDTH, stations=st)
        tst = max(1, math.ceil(U(rng, TEST_MIN) / (takt * PR.LINE_EFF)))
        add(f"test_{a}", f"下线测试 {a + 1}", "line", tst * 4.0 * 5.0 + 30, head=tst, stations=tst)
    total_daily = sum(f["daily"] for f in fams)
    units_per_pallet = int(rng.integers(4, 12))
    n_pack = 2 if tier == "L" else 1
    for k in range(n_pack):
        add(f"pack_{k}", f"包装 {k + 1}", "area", 120 + total_daily / n_pack * 0.6, head=3)
    # ---- stores ----
    rm_pallets = sum(f["daily"] * q / unit[pt]["pallet"] for f in fams for pt, q in f["parts"].items())
    rm_pallets += frames / 6.0
    pp_kit_bins = sum(f["daily"] * f["pp_parts_per_unit"] / 50.0 for f in fams)   # ASM: 50 small parts per bin
    pp_pallets = pp_kit_bins / 24.0                                               # ASM: 24 bins per pallet
    fg_pallets = total_daily / units_per_pallet
    def store_area(pal, days):
        return max(60.0, pal * days * PR.PALLET_FOOT / PR.STACK * PR.STORE_AISLE_FACTOR)
    add("recv", "收货区", "dock", 0, head=3, dock="receiving")
    add("rm", "原材料库", "store", store_area(rm_pallets, PR.INV_DAYS["rm"]), head=3)
    add("pp", "外购件库", "store", store_area(pp_pallets, PR.INV_DAYS["pp"]), head=3)
    sm_bins = sum(f["daily"] * q / unit[pt]["bin"] for f in fams for pt, q in f["parts"].items()) + pp_kit_bins
    for k in range(T["sm"]):
        add(f"sm_{k}", f"线边超市 {k + 1}", "store",
            max(40.0, sm_bins / T["sm"] * PR.INV_DAYS["sm"] * PR.BIN_FOOT / PR.BIN_STACK * 2.5), head=1)
    add("fg", "成品库", "store", store_area(fg_pallets, PR.INV_DAYS["fg"]), head=2)
    add("ship", "发货区", "dock", 0, head=3, dock="shipping")
    add("tool", "刀具室", "support", {"S": 60, "M": 90, "L": 120}[tier], head=1)
    if T["maint"]: add("maint", "设备维修", "support", {"M": 150, "L": 200}[tier], head=2)
    add("qc", "质检室", "support", {"S": 80, "M": 120, "L": 160}[tier], head=2)
    workers = sum(d["headcount_shift"] for d in depts)
    staff = int(round(10 + 0.06 * workers * PR.SHIFTS))
    add("office", "办公室", "support", staff * PR.AREA_PER_OFFICE_STAFF, head=0, staff=staff)
    add("amenity", "更衣室/食堂", "support", max(80.0, workers * PR.AREA_PER_WORKER_AMENITY), head=0)
    doors = {"S": 2, "M": 3, "L": 4}[tier]
    for k in ("recv", "ship"):
        depts[D[k]]["area"] = round(doors * PR.DOCK_CENTER * PR.DOCK_STAGING_DEPTH, 1)
        depts[D[k]]["doors"] = doors
        depts[D[k]]["aspect_max"] = 6.0

    # ---- material flows ----
    for f in fams:
        d = f["daily"]
        sm = f"sm_{f['id'] % T['sm']}"
        fin = f"final_{f['id'] % T['final']}"
        subs = sorted({f"sub_{f['id'] % T['sub']}", f"sub_{(f['id'] + 1) % T['sub']}"})
        for pt, q in f["parts"].items():
            cell = f"cell_{pt}_{f['id'] % T['cells'][pt]}"
            n = d * q
            flow("recv", "rm", n / unit[pt]["pallet"], "forklift", f"{f['name']} {PARTS[pt]['name']}毛坯入库")
            flow("rm", cell, n / unit[pt]["pallet"], "forklift", f"{PARTS[pt]['name']}毛坯")
            b = n / unit[pt]["bin"]
            wash = f"wash_{f['id'] % T['wash']}"
            if PARTS[pt]["heat"]:
                flow(cell, "heat", b, "cart", f"{PARTS[pt]['name']}去热处理")
                flow("heat", cell, b, "cart", f"{PARTS[pt]['name']}热处理后回磨削")
            flow(cell, wash, b, "cart", f"{PARTS[pt]['name']}去清洗")
            flow(wash, sm, b, "tugger", f"{PARTS[pt]['name']}入线边超市")
            flow(sm, subs[0] if pt != "housing" else subs[-1], b, "tugger", f"{PARTS[pt]['name']}上部装线")
        kb = d * f["pp_parts_per_unit"] / 50.0
        flow("recv", "pp", kb / 24.0, "forklift", f"{f['name']} 外购件入库")
        flow("pp", sm, kb, "tugger", f"{f['name']} 外购件补超市")
        flow(sm, fin, kb * 0.6, "tugger", f"{f['name']} 外购件上总装")
        for s in subs:
            flow(sm, s, kb * 0.2, "tugger", f"{f['name']} 外购件上部装")
            flow(s, fin, d / 6.0, "cart", f"{f['name']} 部件上总装")
        if f["frame"]:
            w = f"weld_{f['id'] % max(1, T['weld'])}"
            flow("rm", w, d / 6.0, "forklift", f"{f['name']} 钢材去焊接")
            flow(w, "paint", d / 2.0, "cart", f"{f['name']} 机架去喷漆")
            flow("paint", fin, d / 2.0, "tugger", f"{f['name']} 机架上总装")
        tst = fin.replace("final", "test")
        flow(fin, tst, d, "conveyor", f"{f['name']} 下线测试")
        pk = f"pack_{f['id'] % n_pack}"
        flow(tst, pk, d, "conveyor", f"{f['name']} 去包装")
        flow(pk, "fg", d / units_per_pallet, "forklift", f"{f['name']} 成品入库")
        flow("fg", "ship", d / units_per_pallet, "forklift", f"{f['name']} 发货")
    for d in depts:
        if d["type"] == "cell":
            flow("tool", d["key"], 4 * PR.SHIFTS, "cart", "换刀具")
    # ---- people flows ----
    people = []
    def pf(a, b, trips, what):
        if trips > 0: people.append([a, b, float(trips), what])
    per = sum(PR.PEOPLE_TRIPS.values())
    for d in depts:
        if d["headcount_shift"] > 0:
            pf(d["key"], "amenity", d["headcount_shift"] * per * PR.SHIFTS, "更衣、用餐")
        if d["type"] in ("cell", "monument") and T["maint"]:
            pf("maint", d["key"], 2 * PR.SHIFTS, "设备巡检维修")
        if d["type"] in ("cell", "line"):
            pf("qc", d["key"], 3 * PR.SHIFTS, "首件与巡检")
        if d["type"] == "line":
            pf("office", d["key"], 2 * PR.SHIFTS, "班组长、工艺员")
    pf("office", "amenity", depts[D["office"]]["staff"] * 2, "办公人员用餐")

    # ---- resolve keys ----
    mat = [[D[a], D[b], round(t, 3), eq, what] for a, b, t, eq, what in flows]
    ppl = [[D[a], D[b], round(t, 3), what] for a, b, t, what in people]
    n = len(depts)
    Wm = np.zeros((n, n))
    for i, j, t, eq, _ in mat: Wm[i, j] += t * PR.HANDLING[eq]; Wm[j, i] += t * PR.HANDLING[eq]
    for i, j, t, _ in ppl: Wm[i, j] += t * PR.PEOPLE_RATE; Wm[j, i] += t * PR.PEOPLE_RATE

    # ---- SLP relationship chart (from intensity rank + forced X) ----
    iu = np.triu_indices(n, 1); vals = Wm[iu]; order = np.argsort(-vals); nz = int((vals > 0).sum())
    rel = {}
    cuts = [("A", 0.05), ("E", 0.15), ("I", 0.30), ("O", 0.50)]
    for r, k in enumerate(order[:nz]):
        frac = r / max(1, nz)
        grade = next((g for g, c in cuts if frac < c), "U")
        if grade != "U": rel[(int(iu[0][k]), int(iu[1][k]))] = [grade, "物流/人流强度排序"]
    xs = []
    if "paint" in D:
        for k in [k for k in D if k.startswith("weld") or k == "heat"]:
            xs.append((D["paint"], D[k], "喷漆易燃，远离明火和热源"))
    for k in [k for k in D if k.startswith("cell")]:
        pass
    xs.append((D["qc"], D["heat"], "质检室远离热处理的热和振动"))
    for a, b, why in xs:
        rel[(min(a, b), max(a, b))] = ["X", why]
    relations = [[a, b, g, why] + ([PR.X_GAP] if g == "X" else []) for (a, b), (g, why) in sorted(rel.items())]

    # ---- building ----
    need = sum(d["area"] for d in depts)
    util = 0.70                                                                    # ASM: department area / building area
    nspan = 2 if need < 3500 else 3 if need < 7000 else 4
    H = nspan * PR.SPAN
    W = math.ceil(need / util / H / PR.BAY) * PR.BAY
    if W < 1.3 * H: W = math.ceil(1.3 * H / PR.BAY) * PR.BAY
    cols = [[round(x * PR.BAY, 2), round(y * PR.SPAN, 2)] for x in range(1, int(W / PR.BAY)) for y in range(1, nspan)]
    # main aisle: middle of the middle span (odd spans) or right beside the middle column line (even spans)
    yc = PR.SPAN * (nspan // 2) + PR.SPAN / 2 if nspan % 2 else PR.SPAN * (nspan // 2) + PR.COLUMN / 2 + PR.MAIN_AISLE / 2
    aisles = [dict(id="A0", type="main", traffic=["forklift", "pedestrian"],
                   rect=[0.0, round(yc - PR.MAIN_AISLE / 2, 2), W, PR.MAIN_AISLE])]
    nb = int(W / PR.BAY); ncross = max(1, round(W / 70.0))
    for c in range(ncross):
        bx = round((c + 1) * nb / (ncross + 1)); x = bx * PR.BAY - PR.BAY / 2
        aisles.append(dict(id=f"A{c + 1}", type="cross", traffic=["forklift", "pedestrian"],
                           rect=[round(x - PR.FORKLIFT_AISLE / 2, 2), 0.0, PR.FORKLIFT_AISLE, H]))
    # secondary aisles: split every band deeper than 16 m so that each department can face an aisle
    a_lo0, a_hi0 = yc - PR.MAIN_AISLE / 2, yc + PR.MAIN_AISLE / 2
    for k, (b0, b1) in enumerate([(0.0, a_lo0), (a_hi0, H)]):
        if b1 - b0 > 20.0:
            ym = (b0 + b1) / 2
            aisles.append(dict(id=f"B{k}", type="secondary", traffic=["cart", "pedestrian"],
                               rect=[0.0, round(ym - PR.SECONDARY_AISLE / 2, 2), W, PR.SECONDARY_AISLE]))
    # docks
    # dock doors go into a stretch of wall that no aisle reaches, 2 m clear of the stretch ends
    need_len = (doors - 1) * PR.DOCK_CENTER + PR.DOCK_DOOR_W + 4.0
    def free_stretches(length, cuts):
        pts = sorted([(c[0], c[1]) for c in cuts]); out, s0 = [], 0.0
        for a, b in pts:
            if a > s0: out.append((s0, a))
            s0 = max(s0, b)
        if s0 < length: out.append((s0, length))
        return out
    ycuts = [(a["rect"][1], a["rect"][1] + a["rect"][3]) for a in aisles if a["type"] != "cross"]
    xcuts = [(a["rect"][0], a["rect"][0] + a["rect"][2]) for a in aisles if a["type"] == "cross"]
    def first_center(length, cuts, from_end=False):
        st = [s for s in free_stretches(length, cuts) if s[1] - s[0] >= need_len]
        if not st: raise ValueError("no wall stretch long enough for the dock doors")
        s = st[-1] if from_end else st[0]
        return (s[1] - 2.0 - PR.DOCK_DOOR_W / 2 - (doors - 1) * PR.DOCK_CENTER) if from_end else s[0] + 2.0 + PR.DOCK_DOOR_W / 2
    def doors_on(wall, start, k):
        return [dict(wall=wall, center=round(start + i * PR.DOCK_CENTER, 2), width=PR.DOCK_DOOR_W) for i in range(k)]
    try:
        side_start = first_center(H, ycuts)          # end walls (W/E) if a long enough stretch exists
    except ValueError:
        side_start = None                           # otherwise use the long walls (S/N) near the ends
    if pattern == "I":
        if side_start is not None:
            rd = dict(kind="receiving", doors=doors_on("W", side_start, doors)); sd = dict(kind="shipping", doors=doors_on("E", side_start, doors))
        else:
            rd = dict(kind="receiving", doors=doors_on("S", first_center(W, xcuts), doors)); sd = dict(kind="shipping", doors=doors_on("N", first_center(W, xcuts, True), doors))
    elif pattern == "U":
        rd = dict(kind="receiving", doors=doors_on("S", first_center(W, xcuts), doors)); sd = dict(kind="shipping", doors=doors_on("S", first_center(W, xcuts, True), doors))
    else:  # L
        if side_start is not None:
            rd = dict(kind="receiving", doors=doors_on("W", side_start, doors))
        else:
            rd = dict(kind="receiving", doors=doors_on("N", first_center(W, xcuts), doors))
        sd = dict(kind="shipping", doors=doors_on("S", first_center(W, xcuts, True), doors))
    docks = [rd, sd]
    # exits along both long walls every <= 45 m, plus short walls
    exits = []
    busy = {(dd["wall"], dd["center"]) for dk in docks for dd in dk["doors"]}
    def free(wall, c):
        return all(not (w == wall and abs(c - cc) < PR.DOCK_CENTER) for w, cc in busy)
    nx = max(2, math.ceil(W / 45.0))
    for wall, yy in (("S", 0.0), ("N", H)):
        for k in range(nx):
            c = round((k + 0.5) * W / nx, 2)
            if not free(wall, c): c = round(c + 2 * PR.DOCK_CENTER, 2)
            exits.append(dict(wall=wall, x=c, y=yy, width=PR.EXIT_DOOR_W))
    for wall, xx in (("W", 0.0), ("E", W)):
        for c in (round(H * 0.25, 2), round(H * 0.75, 2)):
            if free(wall, c): exits.append(dict(wall=wall, x=xx, y=c, width=PR.EXIT_DOOR_W))
    fixed = []
    if brownfield:
        # existing buildings sit in corners that are clear of dock doors (door width x staging depth) and of aisles
        zones = []
        for dk in docks:
            for dd in dk["doors"]:
                c, hw, dp = dd["center"], dd["width"] / 2 + 1.0, PR.DOCK_STAGING_DEPTH + 1.0
                zones.append({"W": [0, c - hw, dp, 2 * hw], "E": [W - dp, c - hw, dp, 2 * hw],
                              "S": [c - hw, 0, 2 * hw, dp], "N": [c - hw, H - dp, 2 * hw, dp]}[dd["wall"]])
        zones += [a["rect"] for a in aisles]
        def corner(cn, w, h):
            return {"NE": [W - w, H - h, w, h], "NW": [0.0, H - h, w, h], "SE": [W - w, 0.0, w, h], "SW": [0.0, 0.0, w, h]}[cn]
        def clear(r):
            return all(min(r[0] + r[2], z[0] + z[2]) - max(r[0], z[0]) <= 0 or min(r[1] + r[3], z[1] + z[3]) - max(r[1], z[1]) <= 0 for z in zones)
        o = D["office"]; od = depts[o]
        w = round(math.sqrt(od["area"] * 2), 1); h = math.ceil(od["area"] / w * 10) / 10
        used = None
        for cn in ("NE", "NW", "SE", "SW"):
            r = [round(v, 2) for v in corner(cn, w, h)]
            if clear(r): used = cn; fixed.append(dict(id="F0", dept=o, name="既有办公楼（不可移动）", rect=r)); break
        for cn in ("NW", "SW", "NE", "SE"):
            r = [round(v, 2) for v in corner(cn, 15.0, 12.0)]
            if cn != used and clear(r): fixed.append(dict(id="F1", dept=None, name="空压站/配电房（不可移动）", rect=r)); break
        if fixed and fixed[0]["id"] == "F0": od["fixed_rect"] = fixed[0]["rect"]

    name = f"LA-{tier}-{pattern}{'-BF' if brownfield else ''}-s{seed}"
    inst = dict(
        name=name, version="0.1", kind="assembly-plant", tier=tier, flow_pattern=pattern, brownfield=brownfield, seed=seed,
        units="m", time_basis="trips per working day",
        building=dict(W=W, H=H, span=PR.SPAN, bay=PR.BAY, nspan=nspan, columns=cols, column_size=PR.COLUMN,
                      hazard_class=PR.HAZARD, fire_grade=PR.FIRE_GRADE, sprinklers=PR.SPRINKLER),
        aisles=aisles, docks=docks, exits=exits, fixed=fixed,
        rules=dict(egress_max=PR.EGRESS_MAX, exit_min_spacing=PR.EXIT_MIN_SPACING, compartment_max=PR.COMPARTMENT_MAX,
                   forklift_aisle=PR.FORKLIFT_AISLE, pedestrian_aisle=PR.PED_AISLE, io="edge-midpoint, must open to free floor or aisle"),
        calendar=dict(days=PR.DAYS, shifts=PR.SHIFTS, hours=PR.HOURS, oee=PR.OEE),
        handling=PR.HANDLING, people_rate=PR.PEOPLE_RATE,
        products=[dict(id=f["id"], name=f["name"], annual_volume=f["annual_volume"], frame=f["frame"],
                       machined_parts=f["parts"], purchased_parts_per_unit=f["pp_parts_per_unit"]) for f in fams],
        unit_loads=dict(parts=unit, units_per_pallet=units_per_pallet),
        departments=depts, material_flows=mat, people_flows=ppl, relations=relations,
        summary=dict(n=n, dept_area=round(need, 1), building_area=W * H, utilization=round(need / (W * H), 3),
                     workers_per_shift=workers, office_staff=staff),
    )
    return inst


def main():
    os.makedirs(OUT, exist_ok=True)
    plan = [("S", "I", 1), ("S", "U", 2), ("S", "L", 3),
            ("M", "I", 1), ("M", "U", 2), ("M", "L", 3),
            ("L", "I", 1), ("L", "U", 2), ("L", "L", 3)]
    names = []
    for tier, pat, seed in plan:
        inst = build(tier, pat, seed)
        json.dump(inst, open(os.path.join(OUT, inst["name"] + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        names.append(inst["name"])
    for tier, pat, seed in [("M", "I", 4), ("L", "U", 5)]:
        inst = build(tier, pat, seed, brownfield=True)
        json.dump(inst, open(os.path.join(OUT, inst["name"] + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        names.append(inst["name"])
    json.dump(PR.table(), open(os.path.join(OUT, "_params.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for nm in names:
        s = json.load(open(os.path.join(OUT, nm + ".json"), encoding="utf-8"))["summary"]
        b = json.load(open(os.path.join(OUT, nm + ".json"), encoding="utf-8"))["building"]
        print(nm, s, b["W"], "x", b["H"])


if __name__ == "__main__":
    main()
