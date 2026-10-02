"""Read-only RS4 joint topology discovery; no script/referee state."""
import math


def discover_joint_barriers(stadium, width, height):
    """Accept two red/blue pairs on opposite bands with independent zero-radius anchors.

    This identifies candidates, not active barriers. The real client measures
    their current endpoints; physics does not need to simulate cosmetic joints.
    """
    if not width > 0 or not height > 0:
        return []
    rows = []
    discs = stadium.get("discs", [])
    for number, joint in enumerate(stadium.get("joints", [])):
        color = joint.get("color", -1)
        try:
            color = int(color.lstrip("#"), 16) if isinstance(color, str) else color
        except ValueError:
            continue
        team = 0 if color == 0xEC7458 else 1 if color == 0x48BEF9 else -1
        i, j = joint.get("d0"), joint.get("d1")
        if team < 0 or type(i) is not int or type(j) is not int or not 0 < i < len(discs) or not 0 < j < len(discs) or i == j:
            continue
        a, b = discs[i], discs[j]
        if any(d.get("radius") != 0 or d.get("cMask") not in ([], 0) for d in (a, b)):
            continue
        p, q = a.get("pos"), b.get("pos")
        if not p or not q or len(p) != 2 or len(q) != 2 or not all(math.isfinite(v) for v in (*p, *q)):
            continue
        y, span = (p[1] + q[1]) / 2, abs(p[0] - q[0])
        if abs(p[1] - q[1]) > 3 or not .7 * height <= abs(y) <= 1.2 * height or not 1.5 * width <= span <= 2.5 * width or abs((p[0] + q[0]) / 2) > .1 * width:
            continue
        rows.append((number, team, y, i, j))
    upper, lower = [r for r in rows if r[2] < 0], [r for r in rows if r[2] > 0]
    pair = lambda r: len(r) == 2 and r[0][1] != r[1][1] and abs(r[0][2] - r[1][2]) <= 3
    if not pair(upper) or not pair(lower) or len({d for r in rows for d in r[3:]}) != 8:
        return []
    return [r[0] for r in rows]
