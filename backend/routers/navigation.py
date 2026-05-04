from fastapi import APIRouter, HTTPException
import numpy as np
import heapq

router = APIRouter()

try:
    DOWNSAMPLE = 20
    risk_map  = np.load("outputs/dynamic_risk_map.npy")[::DOWNSAMPLE, ::DOWNSAMPLE]
    slope_map = np.load("outputs/slope.npy")[::DOWNSAMPLE, ::DOWNSAMPLE]
    print(f"Maps loaded: risk={risk_map.shape}, slope={slope_map.shape}")
except Exception as e:
    raise RuntimeError(f"Failed to load maps: {e}")

POLE_MARGIN = 2


def heuristic(a, b, shape):
    h, w  = shape
    lon_a = (a[0] / w) * 2 * np.pi
    lon_b = (b[0] / w) * 2 * np.pi
    lat_a = (a[1] / h) * np.pi
    lat_b = (b[1] / h) * np.pi
    dlat  = lat_b - lat_a
    dlon  = lon_b - lon_a
    aa    = (np.sin(dlat / 2) ** 2
             + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2)
    angle = 2 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
    return angle * (max(h, w) / np.pi)


def get_neighbors(node, shape):
    x, y = node
    h, w  = shape
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx = (x + dx) % w
            ny = y + dy
            if 0 <= ny < h:
                neighbors.append((nx, ny))
    return neighbors


def compute_cost(current, neighbor, risk_map, slope_map, cfg):
    x, y = neighbor
    h, w  = risk_map.shape

    if y < POLE_MARGIN or y >= h - POLE_MARGIN:
        return np.inf

    risk  = risk_map[y, x]
    slope = slope_map[y, x]

    if (slope / 1.2624) > (cfg["max_slope"] / 90.0):
        return np.inf

    lat   = (current[1] / h) * np.pi
    dx    = neighbor[0] - current[0]
    dy    = neighbor[1] - current[1]
    if dx >  w / 2: dx -= w
    if dx < -w / 2: dx += w

    real_dx   = dx * np.cos(lat)
    step_dist = np.sqrt(real_dx ** 2 + dy ** 2)

    return step_dist * (1 + cfg["risk_weight"] * risk + cfg["slope_weight"] * slope)


def astar(start, goal, risk_map, slope_map, cfg):
    open_set = []
    counter  = 0
    heapq.heappush(open_set, (0, counter, start))

    came_from = {}
    g_score   = {start: 0}
    closed    = set()

    while open_set:
        f, _, current = heapq.heappop(open_set)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = []
            total_cost = g_score[current]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, total_cost

        for neighbor in get_neighbors(current, risk_map.shape):
            if neighbor in closed:
                continue
            step_cost = compute_cost(current, neighbor, risk_map, slope_map, cfg)
            if step_cost == np.inf:
                continue
            tentative = g_score[current] + step_cost
            if neighbor not in g_score or tentative < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor]   = tentative
                counter += 1
                heapq.heappush(open_set,
                    (tentative + heuristic(neighbor, goal, risk_map.shape),
                     counter, neighbor))

    return [], np.inf


@router.post("/navigation/path")
def get_path(req: dict):
    try:
        h, w = risk_map.shape

        raw_start = req["start"]
        raw_end   = req["end"]

        grid_start = (int(raw_start[0]), int(raw_start[1]))
        grid_end   = (int(raw_end[0]),   int(raw_end[1]))

        print(f"Display  start={raw_start}  end={raw_end}")
        print(f"Grid     start={grid_start} end={grid_end}")
        print(f"Map shape (h,w)=({h},{w})")

        cfg = req.get("rover_config")
        if not cfg:
            raise HTTPException(status_code=400, detail="Missing rover_config")
        for key in ["max_slope", "risk_weight", "slope_weight"]:
            if key not in cfg:
                raise HTTPException(status_code=400, detail=f"Missing {key}")

        if not (0 <= grid_start[0] < w and 0 <= grid_start[1] < h):
            raise HTTPException(status_code=400, detail="Start out of bounds")
        if not (0 <= grid_end[0] < w and 0 <= grid_end[1] < h):
            raise HTTPException(status_code=400, detail="End out of bounds")

        
        grid_start = (grid_start[0], int(np.clip(grid_start[1], POLE_MARGIN, h - POLE_MARGIN - 1)))
        grid_end   = (grid_end[0],   int(np.clip(grid_end[1],   POLE_MARGIN, h - POLE_MARGIN - 1)))

        path, cost = astar(grid_start, grid_end, risk_map, slope_map, cfg)

        if not path:
            return {"path": [], "steps": 0, "cost": None,
                    "message": "No valid path found"}

        
        display_path = [[p[0], p[1]] for p in path]

        return {
            "path":        display_path,
            "config_used": cfg,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
