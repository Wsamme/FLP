"""Parameters of the lean discrete-assembly benchmark, each with its source.

Every value is tagged:
  SRC  - taken from a public source (code, guideline, textbook); the source is named.
  ASM  - an assumption chosen to look like typical practice. Replace it with project data when available.
"""

P = {}


def p(key, value, tag, note):
    P[key] = dict(value=value, tag=tag, note=note)
    return value


# ---------------- Fire code (GB 50016-2014, 2018 ed.) ----------------
HAZARD = p("building_hazard_class", "丙", "ASM",
           "整栋厂房按丙类（机械装配常见）；喷漆间按 GB 50016 第 3.1.2 条的例外规定不改变整栋类别——该例外的适用条件未逐条核实")
FIRE_GRADE = p("fire_resistance_grade", "二级", "ASM", "钢结构单层厂房常见取二级")
SPRINKLER = p("sprinklers", True, "ASM", "设自动喷水灭火系统")
COMPARTMENT_MAX = p("fire_compartment_max_m2", 16000, "SRC",
                    "GB 50016-2014 表 3.3.1：丙类、二级、单层 8000 m²；第 3.3.3 条设自动灭火系统可增加 1.0 倍 → 16000 m²")
EGRESS_MAX = p("egress_max_m", 80.0, "SRC",
               "GB 50016-2014 表 3.7.4：丙类、一二级、单层厂房任一点到最近安全出口直线距离 80 m")
EXIT_MIN_SPACING = p("exit_min_spacing_m", 5.0, "SRC", "GB 50016-2014 第 3.7.1 条：相邻两个安全出口最近边缘水平距离不应小于 5 m")
EXIT_DOOR_W = p("exit_door_width_m", 1.2, "SRC", "GB 50016-2014 第 3.7.5 条：首层外门净宽不小于 1.20 m")
PED_AISLE = p("pedestrian_aisle_m", 1.4, "SRC", "GB 50016-2014 第 3.7.5 条：疏散走道净宽不小于 1.40 m")

# ---------------- Building grid ----------------
SPAN = p("span_m", 24.0, "ASM",
         "跨度 24 m：单层钢结构厂房常用跨度（GB/T 50006 对跨度和柱距有模数规定，原文未核实）")
BAY = p("bay_m", 9.0, "ASM", "柱距 9 m：常用柱距（同上，未核实原文）")
COLUMN = p("column_size_m", 0.6, "ASM", "柱截面按 0.6 m × 0.6 m 计")

# ---------------- Aisles and docks ----------------
FORKLIFT_AISLE = p("forklift_aisle_m", 3.7, "SRC",
                   "Toyota Forklift 计算方法示例：直角堆垛宽 86 in + 货长 48 in + 间隙 12 in = 146 in ≈ 3.71 m（单台平衡重叉车）")
MAIN_AISLE = p("main_aisle_m", FORKLIFT_AISLE + PED_AISLE, "ASM",
               "主通道 = 叉车道 3.7 m + 并行人行道 1.4 m（人车分道，参照 HSE HSG136 的分离原则）")
SECONDARY_AISLE = p("secondary_aisle_m", 2.5, "ASM", "次通道 2.5 m（手推车和人行；不小于 GB 50016 疏散走道 1.4 m），把超过 20 m 进深的区带一分为二，使每个部门都临通道")
DOCK_DOOR_W = p("dock_door_width_m", 3.0, "SRC", "MHI Dock Planning 101：9 ft 门适合 8 ft 6 in 挂车，10 ft 用于超宽货物；取约 10 ft ≈ 3.0 m")
DOCK_CENTER = p("dock_center_m", 4.3, "SRC", "MHI Dock Planning 101：门中心距最小 12 ft，带门封时常用 14 ft 或更大；取 14 ft ≈ 4.3 m")
DOCK_STAGING_DEPTH = p("dock_staging_depth_m", 15.0, "ASM", "收发货暂存区进深 15 m（MHI 指南未给固定值）")

# ---------------- Calendar and capacity ----------------
DAYS = p("working_days", 250, "ASM", "每年 250 个工作日")
SHIFTS = p("shifts", 2, "ASM", "两班")
HOURS = p("hours_per_shift", 8.0, "ASM", "每班 8 小时（不扣休息）")
OEE = p("oee", 0.85, "ASM", "设备综合效率 85%")
LINE_EFF = p("line_efficiency", 0.90, "ASM", "装配线平衡效率 90%")
STATION_PITCH = p("station_pitch_m", 3.5, "ASM", "装配工位节距 3.5 m")
LINE_WIDTH = p("line_width_m", 6.0, "ASM", "装配线宽 6 m（含两侧物料摆放）")
MACHINE_ALLOW = p("machine_area_allowance", 0.35, "ASM", "设备占地之外再加 35% 作操作与维修空间")
STORE_AISLE_FACTOR = p("store_aisle_factor", 2.0, "ASM", "仓储区面积 = 货位占地 × 2.0（含巷道）")
STACK = p("pallet_stack_levels", 3, "ASM", "货架 3 层")
PALLET_FOOT = p("pallet_footprint_m2", 1.2 * 1.0, "SRC", "ISO 6780 托盘 1200 mm × 1000 mm")
BIN_FOOT = p("bin_footprint_m2", 0.6 * 0.4, "ASM", "周转箱 600 mm × 400 mm")
BIN_STACK = p("bin_stack_levels", 4, "ASM", "流利架 4 层")
INV_DAYS = p("inventory_days", {"rm": 5.0, "pp": 3.0, "sm": 0.5, "fg": 3.0}, "ASM",
             "库存天数：原材料 5 天，外购件 3 天，线边超市 0.5 天，成品 3 天")

# ---------------- Handling equipment (relative cost per trip-metre) ----------------
HANDLING = p("handling_cost_per_trip_m", {"forklift": 1.0, "tugger": 0.35, "cart": 0.6, "conveyor": 0.05}, "ASM",
             "相对单价（每趟每米）：叉车 1.0；牵引车牛奶车 0.35（一趟拖多箱，按每箱分摊）；手推车 0.6；输送线 0.05。没有可靠公开数据，必须用项目数据替换")
PEOPLE_RATE = p("people_cost_per_trip_m", 0.15, "ASM", "人员每趟每米的相对成本 0.15（相对叉车）")
PEOPLE_TRIPS = p("people_trips_per_worker_shift", {"locker": 2, "canteen": 2}, "ASM", "每人每班去更衣室 2 趟、食堂 2 趟（来回各算 1 趟）")

# ---------------- Department shape ----------------
ASPECT_DEFAULT = p("aspect_max_default", 4.0, "ASM", "普通部门最大长宽比 4")
MIN_SIDE = p("min_side_m", 4.0, "ASM", "部门最短边 4 m")
X_GAP = p("x_separation_m", 10.0, "ASM", "X 级（喷漆 vs 焊接/热处理）最小净距 10 m")

# ---------------- Unit areas ----------------
AREA_PER_OFFICE_STAFF = p("office_m2_per_person", 8.0, "ASM", "办公人均 8 m²")
AREA_PER_WORKER_AMENITY = p("amenity_m2_per_worker", 1.2, "ASM", "更衣室和食堂按每名一线员工 1.2 m²（按一个班的人数）")


def table():
    return [dict(key=k, **v) for k, v in P.items()]
