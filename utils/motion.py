# motion layer: small moving things the object detector can't recognise, mostly people and cyclists seen from
# high up, where they are a few pixels across. found with background subtraction, tracked, and labelled by their
# real-world speed, using the cars the detector does see on the same camera as the scale.
# it runs beside the detector, so everything is vectorised: Python-level loops here compete with it for the GIL.
from collections import deque
import cv2
import numpy as np

CAR_LENGTH_M = 4.5
WALK_MAX_MPS = 2.6 # brisk walk / jog
BIKE_MAX_MPS = 9.0 # fast cyclist, anything quicker is a vehicle
MIN_MOVE_MPS = 0.5 # slower than this is drift or noise
STEADY = 0.75 # net movement / path length over the last 1.5s: people go somewhere, noise jitters in place
MAX_SIZE_M = 2.5 # a person or a bike seen from above
MIN_TRAVEL_M = 1.0 # has to actually get somewhere before it is labelled
MAX_BLOBS = 60 # beyond this the frame is mostly noise (rain, a flashing screen), keep the smallest
LOST_AFTER = 1.0 # seconds without a matching blob before a track ends
REATTACH_WITHIN = 8.0 # an ended track can be taken over by a new blob for this long, so nobody is counted twice
REATTACH_MAX_PX = 60 # but never further than this from where it should be, or boxes hop onto unrelated blobs
LANE_CELL = 16 # px, resolution of the learned car lane map
PAN_FRACTION = 0.12 # more of the picture than this changing at once means the camera itself moved (PTZ cams)
ID_OFFSET = 1_000_000 # keeps motion track ids apart from detector track ids

class MotionDetector:
  def __init__(self, width=1280):
    self.width = width
    self.bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=32, detectShadows=False)
    self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    self.px_per_m = None
    self.car_lengths = deque(maxlen=400)
    # tracks as parallel arrays: id, centre, velocity (px/s), size (w, h), last time matched, matches so far
    self.next_id = 1
    self.ids, self.hits = np.zeros(0, np.int64), np.zeros(0, np.int64)
    self.pos, self.vel, self.wh = np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
    self.t_last = np.zeros(0)
    self.trails = {} # track id -> deque of (t, cx, cy)
    self.labels = {} # track id -> "pedestrian" / "cyclist" once classified
    self.lost = {} # recently ended tracks that a new blob can still take over: id -> (pos, vel, wh, t_last, hits)
    self.lanes, self.lane_mask, self.lane_updates = None, None, 0 # where the detector keeps seeing vehicles
    self.panning = False

  def learn_scale(self, car_lengths_px):
    # cars are ~4.5m long, their median size in pixels gives metres per pixel for this camera. the median runs over
    # the last few hundred sightings, not per frame: with two or three cars in a frame one odd box (part of a tram)
    # would drag it far off, and every speed with it
    self.car_lengths.extend(car_lengths_px)
    if len(self.car_lengths) >= 20: self.px_per_m = float(np.median(self.car_lengths)) / CAR_LENGTH_M

  def learn_lanes(self, vehicle_boxes, W, H):
    # heatmap of where cars, buses and trucks drive. anything at cycling speed in there is a car part the detector
    # missed, which is the main way a car would otherwise end up labelled a cyclist
    if self.lanes is None: self.lanes = np.zeros((H // LANE_CELL + 1, W // LANE_CELL + 1), np.float32)
    self.lanes *= 0.9995
    for x1, y1, x2, y2 in vehicle_boxes: # middle half of each box, so lanes don't bleed onto bike paths next to them
      cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 4, (y2 - y1) / 4
      self.lanes[int((cy - hh) // LANE_CELL):int((cy + hh) // LANE_CELL) + 1, int((cx - hw) // LANE_CELL):int((cx + hw) // LANE_CELL) + 1] += 1
    self.lane_updates += 1
    if self.lane_updates % 20 == 0:
      # a lane is anywhere a vehicle covered at least 2% of the time; relative to the busiest cell would let parked
      # cars (there every frame) set the bar so high that real traffic lanes never count
      frames = (1 - 0.9995 ** self.lane_updates) / (1 - 0.9995)
      self.lane_mask = self.lanes >= max(2.0, 0.02 * frames)

  def in_lane(self, x, y):
    if self.lane_mask is None: return False
    r, c = int(y // LANE_CELL), int(x // LANE_CELL)
    return 0 <= r < self.lane_mask.shape[0] and 0 <= c < self.lane_mask.shape[1] and bool(self.lane_mask[r, c])

  def update(self, frame, t, known_boxes):
    """frame: BGR full res, t: stream time in seconds, known_boxes: [x1,y1,x2,y2] the detector already explains.
    returns [(track_id, label, box, speed_mps)] for tracks classified as pedestrian or cyclist"""
    cents, wh = self.blobs(frame, known_boxes)
    if self.panning: # everything shifts while the camera moves: drop what we had and start over once it settles
      self.reset_tracks()
      return []
    self.track(cents, wh, t)
    out = []
    for i in np.flatnonzero((self.t_last == t) & (self.hits >= 3)):
      tid = int(self.ids[i])
      trail = self.trails.setdefault(tid, deque())
      trail.append((t, *self.pos[i]))
      while t - trail[0][0] > 2.0: trail.popleft()
      label = self.classify(tid, trail, float(self.wh[i].max()))
      if label:
        (cx, cy), (w, h) = self.pos[i], self.wh[i]
        out.append((ID_OFFSET + tid, label, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], self.speed(trail)))
    return out

  def blobs(self, frame, known_boxes):
    H, W = frame.shape[:2]
    k = self.width / W
    small = cv2.resize(frame, (self.width, int(H * k)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    # while the camera moves, learn the new view fast instead of over the usual few hundred frames
    mask = self.bg.apply(gray, learningRate=0.2 if self.panning else -1)
    self.panning = cv2.countNonZero(mask) > PAN_FRACTION * mask.size
    if self.panning: return np.zeros((0, 2)), np.zeros((0, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
    _, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats, cents = stats[1:], cents[1:] / k
    max_px = (MAX_SIZE_M * self.px_per_m * k) if self.px_per_m else 30
    keep = (stats[:, 4] >= 8) & (np.maximum(stats[:, 2], stats[:, 3]) <= max_px)
    if len(known_boxes) and keep.any(): # skip what the detector already boxed
      kb = np.asarray(known_boxes, dtype=np.float64)
      pad = 0.25 * (kb[:, 2:4] - kb[:, 0:2])
      lo, hi = kb[:, 0:2] - pad, kb[:, 2:4] + pad
      keep &= ~((cents[:, None, :] >= lo[None]) & (cents[:, None, :] <= hi[None])).all(-1).any(-1)
    idx = np.flatnonzero(keep)
    if len(idx) > MAX_BLOBS: idx = idx[np.argsort(stats[idx, 4])[:MAX_BLOBS]]
    return cents[idx], stats[idx, 2:4] / k

  def reset_tracks(self):
    self.ids, self.hits = np.zeros(0, np.int64), np.zeros(0, np.int64)
    self.pos, self.vel, self.wh = np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
    self.t_last = np.zeros(0)
    self.trails.clear(); self.labels.clear(); self.lost.clear()

  def track(self, cents, wh, t):
    # match blobs to where each track should be now (constant velocity), nearest first, within a size based gate
    match = np.full(len(cents), -1)
    if len(self.ids) and len(cents):
      pred = self.pos + self.vel * (t - self.t_last)[:, None]
      d = np.linalg.norm(cents[:, None, :] - pred[None], axis=-1)
      gate = np.maximum(12.0, 2.5 * np.maximum(wh.max(1)[:, None], self.wh.max(1)[None]))
      d[d > gate] = np.inf
      for _ in range(min(d.shape)):
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if not np.isfinite(d[i, j]): break
        match[i] = j
        d[i, :], d[:, j] = np.inf, np.inf
    got = match >= 0
    j = match[got]
    if len(j):
      v = (cents[got] - self.pos[j]) / np.maximum(t - self.t_last[j], 1e-3)[:, None]
      self.vel[j] = np.where(self.hits[j, None] > 1, 0.6 * self.vel[j] + 0.4 * v, v)
      self.pos[j], self.wh[j], self.t_last[j] = cents[got], wh[got], t
      self.hits[j] += 1
    n = int((~got).sum())
    if n:
      new_c, new_wh = cents[~got], wh[~got]
      ids, vel, hits = np.arange(self.next_id, self.next_id + n), np.zeros((n, 2)), np.ones(n, np.int64)
      for i, lid in self.reattach(new_c, new_wh, t).items(): # same person back after a tree, a tram, a lamp post
        lp, lv, lwh, lt, lh = self.lost.pop(lid)
        ids[i], vel[i], hits[i] = lid, lv, lh + 1
      fresh = ids >= self.next_id
      ids[fresh] = np.arange(self.next_id, self.next_id + int(fresh.sum())); self.next_id += int(fresh.sum())
      self.ids, self.pos, self.wh = np.concatenate([self.ids, ids]), np.vstack([self.pos, new_c]), np.vstack([self.wh, new_wh])
      self.vel = np.vstack([self.vel, vel])
      self.t_last, self.hits = np.concatenate([self.t_last, np.full(n, t)]), np.concatenate([self.hits, hits])
    alive = t - self.t_last <= LOST_AFTER
    if not alive.all():
      for i in np.flatnonzero(~alive): # established tracks can still be taken over for a while, the rest just end
        tid = int(self.ids[i])
        if self.hits[i] >= 3: self.lost[tid] = (self.pos[i].copy(), self.vel[i].copy(), self.wh[i].copy(), float(self.t_last[i]), int(self.hits[i]))
        else: self.trails.pop(tid, None); self.labels.pop(tid, None)
      self.ids, self.hits, self.pos, self.vel, self.wh, self.t_last = (a[alive] for a in (self.ids, self.hits, self.pos, self.vel, self.wh, self.t_last))
    for tid in [tid for tid, v in self.lost.items() if t - v[3] > REATTACH_WITHIN]:
      del self.lost[tid]; self.trails.pop(tid, None); self.labels.pop(tid, None)

  def reattach(self, cents, wh, t):
    """match new blobs to recently lost tracks: where the track would be by now at its last velocity,
    within a radius that grows with how long it was gone and how fast it moved. returns {blob index: lost id}"""
    if not self.lost or not len(cents): return {}
    lids = [i for i in self.lost if i in self.labels] # only real, labelled tracks get taken over
    if not lids: return {}
    lp = np.array([self.lost[i][0] for i in lids]); lv = np.array([self.lost[i][1] for i in lids])
    lwh = np.array([self.lost[i][2] for i in lids]); lt = np.array([self.lost[i][3] for i in lids])
    gone = (t - lt)[None, :]
    d = np.linalg.norm(cents[:, None, :] - (lp + lv * (t - lt)[:, None])[None], axis=-1)
    radius = np.minimum(15.0 + 1.0 * np.linalg.norm(lv, axis=1)[None, :] * gone + 2.0 * lwh.max(1)[None, :], REATTACH_MAX_PX) # room to turn a corner
    size_ok = np.abs(wh.max(1)[:, None] - lwh.max(1)[None, :]) <= 1.5 * lwh.max(1)[None, :] + 4
    d[(d > radius) | ~size_ok] = np.inf
    out = {}
    for _ in range(min(d.shape)):
      i, j = np.unravel_index(np.argmin(d), d.shape)
      if not np.isfinite(d[i, j]): break
      out[int(i)] = lids[j]
      d[i, :], d[:, j] = np.inf, np.inf
    return out

  def speed(self, trail):
    if not self.px_per_m or len(trail) < 2 or trail[-1][0] - trail[0][0] <= 0: return 0.0
    (t0, x0, y0), (t1, x1, y1) = trail[0], trail[-1]
    return float(np.hypot(x1 - x0, y1 - y0) / (t1 - t0) / self.px_per_m)

  def classify(self, tid, trail, size_px):
    # judged on the last 1.5s every frame: a label only shows while the thing is actually walking or riding,
    # so nothing stays stuck on a tree, a reflection or an empty road
    if not self.px_per_m or trail[-1][0] - trail[0][0] < 1.5 or size_px / self.px_per_m > MAX_SIZE_M: return None
    p = np.asarray([q for q in trail if trail[-1][0] - q[0] <= 1.5])[:, 1:]
    path = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    net = np.linalg.norm(p[-1] - p[0])
    v = self.speed(trail)
    if path <= 0 or net / path < STEADY or net / self.px_per_m < MIN_TRAVEL_M or not MIN_MOVE_MPS <= v < BIKE_MAX_MPS: return None
    # both halves of the window must head the same way and each cover real ground: wobbling things reverse
    mid = p[len(p) // 2]
    a, b = mid - p[0], p[-1] - mid
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if min(na, nb) < 0.3 * net or float(a @ b) < 0.7 * na * nb: return None
    if self.in_lane(*p[-1]): return None # in a car lane: a missed piece of a vehicle far more often than a person
    label = "pedestrian" if v < WALK_MAX_MPS else "cyclist"
    if self.labels.get(tid) == "cyclist" and v >= MIN_MOVE_MPS * 2: label = "cyclist" # a rider slowing down stays one
    self.labels[tid] = label
    return label
