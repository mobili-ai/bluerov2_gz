#!/usr/bin/env python3
"""Generate the Newport News Shipbuilding (HII) waterfront world.

Real-scale, approximate reconstruction of the yard on the east bank of the James River:
sheet-pile bulkhead, pile-supported finger piers, two graving docks with a goliath
gantry crane, a moored carrier and two submarines, work craft, and a dredged, littered
seabed. Geometry is built in a "yard frame" (x inland / -x into the river, y along the
shore towards the north end of the yard) and the whole yard is yawed so the shoreline
runs NNW like the real one.

Writes:
  models/nns_shipyard/meshes/*.obj   (meshes, vertices already in yard frame)
  worlds/nns_shipyard.world

Usage: python3 scripts/generate_nns_world.py
"""

import math
import os
import shutil

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PKG, 'models', 'nns_shipyard')
MESH_DIR = os.path.join(MODEL_DIR, 'meshes')
WORLD_PATH = os.path.join(PKG, 'worlds', 'nns_shipyard.world')
MESH_URI = 'model://nns_shipyard/meshes'

# Shoreline bearing ~332 deg true: yard +y points 28 deg west of north.
YARD_YAW = math.radians(28.0)
ORIGIN_LAT, ORIGIN_LON = 36.9850, -76.4440

LAND_Z = 3.0
LAND_DEPTH_X = 450.0
SHORE_Y = (-700.0, 700.0)
RIVER_X = -900.0

# (name, y_center, length, width)
PIERS = [
    ('pier_1', 330.0, 300.0, 30.0),
    ('pier_2', 200.0, 260.0, 25.0),
    ('pier_3', 20.0, 340.0, 36.0),
    ('pier_4', -130.0, 230.0, 24.0),
    ('pier_5', -260.0, 260.0, 26.0),
    ('pier_6', -420.0, 220.0, 22.0),
    ('pier_7', -560.0, 180.0, 20.0),
]
DECK_TOP = 3.4
DECK_THICK = 0.9
CAP_DEPTH = 0.9
BENT_SPACING = 6.0
PILE_SPACING_Y = 4.5
PILE_SIZE = 0.6
FENDER_PILE_R = 0.18

# (name, y_min, y_max, length_inland, flooded)
DOCKS = [
    ('dock_a', 430.0, 480.0, 300.0, True),
    ('dock_b', 540.0, 610.0, 320.0, False),
]
DOCK_FLOOR = -12.5
DOCK_WALL = 3.0

CARRIER = dict(x_mid=-180.0, y=-22.9, L=317.0, B=40.8, T=12.0, D=18.0)
SUBS = [dict(name='ssn_north', x_mid=-100.0, y=220.7), dict(name='ssn_south', x_mid=-120.0, y=-238.8)]

# Off the end of pier 3, facing the shore: pier piles ahead, carrier bulb to starboard.
ROV_SPAWN_YARD = (-352.0, 0.0, -4.0, 0.0)   # x, y, z, yaw (yard frame)

RNG = np.random.default_rng(20261117)


# --------------------------------------------------------------------------- colours
GREY_HAZE = (0.42, 0.45, 0.48)
ANTIFOUL = (0.32, 0.10, 0.09)
SUB_BLACK = (0.04, 0.04, 0.045)
CONCRETE = (0.58, 0.57, 0.54)
CONCRETE_WET = (0.36, 0.37, 0.35)
STEEL_RUST = (0.33, 0.22, 0.16)
TIMBER = (0.30, 0.24, 0.17)
ASPHALT = (0.25, 0.25, 0.26)
CRANE_RED = (0.70, 0.10, 0.08)
CRANE_YELLOW = (0.85, 0.65, 0.10)
WHITE = (0.85, 0.85, 0.85)

# Acoustic reflectivity in [0,1], read by the sonar sim through <laser_retro>.
RETRO = dict(steel=0.9, concrete=0.7, timber=0.45, rock=0.6, mud=0.3, anechoic=0.2,
             rubber=0.35, debris=0.65)


# --------------------------------------------------------------------------- mesh
class Mesh:

    def __init__(self):
        self.v = []
        self.uv = []
        self.f = []

    def _add(self, verts, faces, uvs=None, ref=None):
        """Append triangles. `ref` (a point, or one point per vertex) lies on the inside of
        the surface; faces are flipped so their normals point away from it, because
        ogre2 culls back faces."""
        verts = np.asarray(verts, float)
        faces = np.asarray(faces, int).reshape(-1, 3)
        if ref is not None and len(faces):
            ref = np.broadcast_to(np.asarray(ref, float), verts.shape)
            a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            n = np.cross(b - a, c - a)
            out = (a + b + c) / 3 - (ref[faces[:, 0]] + ref[faces[:, 1]] + ref[faces[:, 2]]) / 3
            flip = np.einsum('ij,ij->i', n, out) < 0
            faces[flip] = faces[flip][:, [0, 2, 1]]
        base = len(self.v)
        self.v.extend(map(tuple, verts))
        if uvs is not None:
            self.uv.extend(map(tuple, uvs))
        self.f.extend(map(tuple, faces + base))

    def grid(self, pts, uvs=None, wrap_u=False, ref=None):
        """pts: (nu, nv, 3) surface grid, quads between neighbours."""
        nu, nv = pts.shape[:2]
        idx = np.arange(nu * nv).reshape(nu, nv)
        i = np.arange(nu if wrap_u else nu - 1)
        i2 = (i + 1) % nu
        j = np.arange(nv - 1)
        a, b = idx[i][:, j], idx[i2][:, j]
        c, d = idx[i2][:, j + 1], idx[i][:, j + 1]
        faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
        if ref is not None and np.ndim(ref) > 1:
            ref = np.asarray(ref).reshape(-1, 3)
        self._add(pts.reshape(-1, 3), faces, None if uvs is None else uvs.reshape(-1, 2), ref)

    def fan(self, ring, center, ref):
        pts = [center] + list(ring)
        faces = [(0, i + 1, (i + 1) % len(ring) + 1) for i in range(len(ring))]
        self._add(pts, faces, ref=ref)

    def prism(self, p0, p1, radius, sides=8, cap=True, phase=0.0):
        """Prism of `sides` from p0 to p1 (round pile, chain link, shaft...)."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        axis = p1 - p0
        axis /= np.linalg.norm(axis)
        up = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(axis, up)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        ang = phase + np.linspace(0, 2 * np.pi, sides, endpoint=False)
        ring = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
        pts = np.stack([p0 + radius * ring, p1 + radius * ring], axis=1)
        ref = np.stack([np.broadcast_to(p0, ring.shape), np.broadcast_to(p1, ring.shape)], axis=1)
        self.grid(pts, wrap_u=True, ref=ref)
        if cap:
            mid = (p0 + p1) / 2
            self.fan(p1 + radius * ring, p1, mid)
            self.fan(p0 + radius * ring, p0, mid)

    def box(self, center, size, yaw=0.0):
        c = np.asarray(center, float)
        hx, hy, hz = np.asarray(size, float) / 2
        corners = np.array([[sx * hx, sy * hy, sz * hz] for sz in (-1, 1)
                            for sy in (-1, 1) for sx in (-1, 1)])
        cy, sy_ = math.cos(yaw), math.sin(yaw)
        rot = np.array([[cy, -sy_, 0], [sy_, cy, 0], [0, 0, 1]])
        corners = corners @ rot.T + c
        quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
        for q in quads:
            self._add(corners[list(q)], [(0, 1, 2), (0, 2, 3)], ref=c)

    def ellipsoid(self, center, radii, n=12):
        th = np.linspace(0, np.pi, n)
        ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
        T, P = np.meshgrid(th, ph, indexing='ij')
        pts = np.stack([np.cos(T) * radii[0], np.sin(T) * np.cos(P) * radii[1],
                        np.sin(T) * np.sin(P) * radii[2]], axis=-1) + np.asarray(center)
        self.grid(pts.transpose(1, 0, 2), wrap_u=True, ref=center)

    def torus(self, center, R, r, normal_axis='z', n=16, m=8):
        u = np.linspace(0, 2 * np.pi, n, endpoint=False)
        v = np.linspace(0, 2 * np.pi, m + 1)
        U, V = np.meshgrid(u, v, indexing='ij')
        pts = np.stack([(R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), r * np.sin(V)], -1)
        tube = np.stack([R * np.cos(U), R * np.sin(U), np.zeros_like(U)], -1)
        order = {'x': [2, 0, 1], 'y': [0, 2, 1], 'z': [0, 1, 2]}[normal_axis]
        c = np.asarray(center)
        self.grid(pts[..., order] + c, wrap_u=True, ref=tube[..., order] + c)

    def rock(self, center, size):
        """Perturbed icosahedron."""
        t = (1 + 5 ** 0.5) / 2
        v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                      [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v *= RNG.uniform(0.7, 1.1, (12, 1))
        v *= np.array([1.0, RNG.uniform(0.7, 1.2), RNG.uniform(0.5, 0.8)]) * size / 2
        faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
                 (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
                 (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
        self._add(v + np.asarray(center), faces, ref=center)

    def write(self, name):
        path = os.path.join(MESH_DIR, name + '.obj')
        v = np.asarray(self.v)
        f = np.asarray(self.f)
        normals = np.zeros_like(v)
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for k in range(3):
            np.add.at(normals, f[:, k], fn)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        with open(path, 'w') as fh:
            fh.write(f'# generated by generate_nns_world.py\no {name}\n')
            fh.write(''.join(f'v {a:.3f} {b:.3f} {c:.3f}\n' for a, b, c in v))
            fh.write(''.join(f'vn {a:.4f} {b:.4f} {c:.4f}\n' for a, b, c in normals))
            if self.uv:
                fh.write(''.join(f'vt {a:.4f} {b:.4f}\n' for a, b in self.uv))
                fh.write(''.join(f'f {a+1}/{a+1}/{a+1} {b+1}/{b+1}/{b+1} {c+1}/{c+1}/{c+1}\n'
                                 for a, b, c in f))
            else:
                fh.write(''.join(f'f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n' for a, b, c in f))
        return f'{MESH_URI}/{name}.obj', len(f)


# --------------------------------------------------------------------------- SDF helpers
def _pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    return f'<pose>{x:.3f} {y:.3f} {z:.3f} {roll:.4f} {pitch:.4f} {yaw:.4f}</pose>'


def _material(color, specular=0.1, alpha=1.0, albedo_map=None, double_sided=False):
    r, g, b = color
    pbr = ''
    if albedo_map:
        pbr = (f'<pbr><metal><albedo_map>{albedo_map}</albedo_map><roughness>0.95</roughness>'
               f'<metalness>0.0</metalness></metal></pbr>')
    ds = '<double_sided>true</double_sided>' if double_sided else ''
    return (f'<material><ambient>{r*0.6:.3f} {g*0.6:.3f} {b*0.6:.3f} {alpha}</ambient>'
            f'<diffuse>{r:.3f} {g:.3f} {b:.3f} {alpha}</diffuse>'
            f'<specular>{specular} {specular} {specular} 1</specular>{ds}{pbr}</material>')


class Link:

    def __init__(self, name):
        self.name = name
        self.items = []
        self._n = 0

    def _uid(self, prefix):
        self._n += 1
        return f'{prefix}_{self._n}'

    def add(self, geometry, pose, color, retro, collide=True, **mat):
        name = self._uid('v')
        retro_xml = f'<laser_retro>{retro}</laser_retro>' if retro is not None else ''
        self.items.append(f'<visual name="{name}">{pose}<geometry>{geometry}</geometry>'
                          f'{_material(color, **mat)}{retro_xml}</visual>')
        if collide:
            self.items.append(f'<collision name="{name}_c">{pose}<geometry>{geometry}</geometry></collision>')

    def box(self, center, size, color, retro, yaw=0.0, collide=True, roll=0.0, pitch=0.0, **mat):
        g = f'<box><size>{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}</size></box>'
        self.add(g, _pose(*center, roll, pitch, yaw), color, retro, collide, **mat)

    def cylinder(self, center, radius, length, color, retro, roll=0.0, pitch=0.0, yaw=0.0,
                 collide=True, **mat):
        g = f'<cylinder><radius>{radius:.3f}</radius><length>{length:.3f}</length></cylinder>'
        self.add(g, _pose(*center, roll, pitch, yaw), color, retro, collide, **mat)

    def mesh(self, uri, color, retro, collide=True, **mat):
        g = f'<mesh><uri>{uri}</uri></mesh>'
        self.add(g, _pose(0, 0, 0), color, retro, collide, **mat)

    def xml(self):
        return f'<link name="{self.name}">' + ''.join(self.items) + '</link>'


def model_xml(name, links, label, pose=(0, 0, 0, 0, 0, YARD_YAW)):
    return (f'<model name="{name}"><static>true</static>{_pose(*pose)}'
            + ''.join(l.xml() for l in links)
            + f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
              f'<label>{label}</label></plugin></model>')


# --------------------------------------------------------------------------- seabed
def in_carrier_berth(x, y):
    c = CARRIER
    dx = np.maximum(np.abs(x - c['x_mid']) - (c['L'] / 2 + 20), 0)
    dy = np.maximum(np.abs(y - c['y']) - (c['B'] / 2 + 6), 0)
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * 15.0 ** 2))


def seabed_depth(x, y):
    """Positive depth below the water surface at yard (x, y)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    berth = 13.7
    channel = 15.0
    d = np.where(x > -380, berth, np.interp(x, [-900, -650, -380], [7.0, channel, berth]))
    toe = np.clip(-x / 25.0, 0, 1)
    d = 11.0 + (d - 11.0) * (3 * toe ** 2 - 2 * toe ** 3)
    d = d + (15.4 - berth) * in_carrier_berth(x, y) * (x > -380)
    for _, yc, length, width in PIERS:
        r2 = (x + length) ** 2 + (y - yc) ** 2
        d = d + 1.2 * np.exp(-r2 / (2 * 14.0 ** 2))
    k = RNG_SEABED
    for amp, wl, ang, ph in k:
        d = d + amp * np.sin(2 * np.pi * (x * math.cos(ang) + y * math.sin(ang)) / wl + ph)
    sandwave = 0.18 * np.sin(2 * np.pi * (x * 0.95 + y * 0.31) / 9.0) * np.clip((-x - 300) / 100, 0, 1)
    return d + sandwave


RNG_SEABED = [(RNG.uniform(0.1, 0.45), RNG.uniform(25, 160), RNG.uniform(0, np.pi), RNG.uniform(0, 6.3))
              for _ in range(10)]


def build_seabed():
    xs = np.linspace(RIVER_X, 1.0, 226)
    ys = np.linspace(SHORE_Y[0], SHORE_Y[1], 351)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    Z = -seabed_depth(X, Y)
    pts = np.stack([X, Y, Z], -1)
    uvs = np.stack([X / 6.0, Y / 6.0], -1)
    m = Mesh()
    m.grid(pts, uvs, ref=pts - [0, 0, 10.0])
    return m.write('seabed')


def ground_z(x, y):
    return float(-seabed_depth(x, y))


# --------------------------------------------------------------------------- bulkhead + land
def dock_gaps():
    return [(y0, y1) for _, y0, y1, _, _ in DOCKS]


def build_bulkhead():
    """Trapezoidal sheet-pile wall along x=0 with gaps at the dock entrances."""
    m = Mesh()
    pitch, depth = 1.2, 0.35
    profile = [(0.0, 0.0), (0.25, 0.0), (0.45, depth), (0.75, depth), (0.95, 0.0)]
    segments = []
    y = SHORE_Y[0]
    for y0, y1 in sorted(dock_gaps()):
        segments.append((y, y0))
        y = y1
    segments.append((y, SHORE_Y[1]))
    for s0, s1 in segments:
        n = int((s1 - s0) / pitch)
        line = [(s0 + i * pitch + py * pitch, px) for i in range(n) for py, px in profile]
        line.append((s0 + n * pitch, 0.0))
        line = np.array(line)
        pts = np.zeros((len(line), 2, 3))
        pts[:, 0] = np.c_[line[:, 1], line[:, 0], np.full(len(line), -17.0)]
        pts[:, 1] = np.c_[line[:, 1], line[:, 0], np.full(len(line), 2.2)]
        m.grid(pts, ref=pts + [5.0, 0, 0])
    return m.write('bulkhead'), segments


def build_land_and_docks(link_land, link_dock, link_bulkhead_cap, segments):
    for s0, s1 in segments:
        link_bulkhead_cap.box(((0.45), (s0 + s1) / 2, 2.7), (1.5, s1 - s0, 1.4), CONCRETE, RETRO['concrete'])
        link_land.box((LAND_DEPTH_X / 2 + 0.4, (s0 + s1) / 2, (LAND_Z - 17.0) / 2),
                      (LAND_DEPTH_X, s1 - s0, LAND_Z + 17.0), ASPHALT, RETRO['concrete'])
    for name, y0, y1, length, flooded in DOCKS:
        w = y1 - y0
        yc = (y0 + y1) / 2
        behind = LAND_DEPTH_X - length
        link_land.box((length + behind / 2 + 0.4, yc, (LAND_Z - 17.0) / 2), (behind, w, LAND_Z + 17.0),
                      ASPHALT, RETRO['concrete'])
        floor_c = CONCRETE_WET if flooded else CONCRETE
        link_dock.box((length / 2, yc, DOCK_FLOOR - 1.0), (length, w, 2.0), floor_c, RETRO['concrete'])
        for side in (-1, 1):
            wy = yc + side * (w / 2 - DOCK_WALL / 2)
            link_dock.box((length / 2, wy, (DOCK_FLOOR + LAND_Z) / 2), (length, DOCK_WALL, LAND_Z - DOCK_FLOOR),
                          floor_c, RETRO['concrete'])
            for k, (zt, ww) in enumerate([(-8.0, 2.0), (-4.0, 1.4)]):
                ay = yc + side * (w / 2 - DOCK_WALL - ww / 2)
                link_dock.box((length / 2 + 6, ay, (DOCK_FLOOR + zt) / 2), (length - 12, ww, zt - DOCK_FLOOR),
                              floor_c, RETRO['concrete'])
        link_dock.box((length - DOCK_WALL / 2, yc, (DOCK_FLOOR + LAND_Z) / 2),
                      (DOCK_WALL, w, LAND_Z - DOCK_FLOOR), floor_c, RETRO['concrete'])
        link_dock.box((2.0, yc, DOCK_FLOOR + 0.6), (4.0, w - 2 * DOCK_WALL, 1.2), CONCRETE_WET, RETRO['concrete'])
        # Keel and bilge blocks.
        for bx in np.arange(30.0, length - 30.0, 4.5):
            link_dock.box((bx, yc, DOCK_FLOOR + 0.8), (1.2, 2.4, 1.6), TIMBER, RETRO['timber'])
            if int(bx) % 3 == 0:
                for side in (-1, 1):
                    link_dock.box((bx, yc + side * 9.0, DOCK_FLOOR + 0.6), (1.2, 1.8, 1.2),
                                  TIMBER, RETRO['timber'])
        if flooded:
            # Hull module (superlift) resting on the blocks.
            link_dock.box((140.0, yc, DOCK_FLOOR + 1.6 + 5.0), (36.0, 30.0, 10.0), STEEL_RUST, RETRO['steel'])
            link_dock.box((140.0, yc, DOCK_FLOOR + 1.6 + 10.5), (36.0, 30.0, 1.0), GREY_HAZE, RETRO['steel'])
        else:
            # Closed caisson gate seated at the entrance.
            link_dock.box((-1.5, yc, (DOCK_FLOOR + LAND_Z) / 2), (6.0, w, LAND_Z - DOCK_FLOOR + 0.5),
                          STEEL_RUST, RETRO['steel'])
    # Floating caisson for the flooded dock, moored alongside the bulkhead.
    y0, _ = DOCKS[0][1], DOCKS[0][2]
    link_dock.box((-6.5, y0 - 30.0, -3.0), (6.0, 50.0, 12.0), STEEL_RUST, RETRO['steel'])


# --------------------------------------------------------------------------- piers
def build_pier(name, yc, length, width, link):
    piles = Mesh()
    fenders = Mesh()
    y_min, y_max = yc - width / 2, yc + width / 2
    link.box((-length / 2, yc, DECK_TOP - DECK_THICK / 2), (length, width, DECK_THICK), CONCRETE,
             RETRO['concrete'])
    cap_z = DECK_TOP - DECK_THICK - CAP_DEPTH / 2
    pile_top = DECK_TOP - DECK_THICK - CAP_DEPTH
    n_across = max(int(round(width / PILE_SPACING_Y)), 2)
    ys = np.linspace(y_min + 1.0, y_max - 1.0, n_across + 1)
    bents = np.arange(-4.0, -length, -BENT_SPACING)
    for i, bx in enumerate(bents):
        link.box((bx, yc, cap_z), (1.0, width, CAP_DEPTH), CONCRETE, RETRO['concrete'])
        for j, py in enumerate(ys):
            bottom = ground_z(bx, py) - 4.0
            if i % 4 == 2 and j in (0, len(ys) - 1):
                lean = -1 if j == 0 else 1
                piles.prism((bx, py + lean * 3.0, bottom), (bx, py, pile_top), PILE_SIZE / 2, sides=8,
                            phase=np.pi / 8)
            piles.prism((bx, py, bottom), (bx, py, pile_top), PILE_SIZE / 2, sides=8, phase=np.pi / 8)
    # Timber fender system: piles + two wales on both long sides and the outer end.
    edge_pts = [(x, y_min - 0.6) for x in np.arange(-3.0, -length, -3.0)]
    edge_pts += [(x, y_max + 0.6) for x in np.arange(-3.0, -length, -3.0)]
    edge_pts += [(-length - 0.6, y) for y in np.arange(y_min, y_max, 3.0)]
    for fx, fy in edge_pts:
        lean = RNG.normal(0, 0.05, 2)
        fenders.prism((fx + lean[0], fy + lean[1], ground_z(fx, fy) - 2.0), (fx, fy, DECK_TOP + 0.4),
                      FENDER_PILE_R, sides=7)
    for wz in (0.8, 2.4):
        for wy in (y_min - 0.85, y_max + 0.85):
            link.box((-length / 2, wy, wz), (length, 0.3, 0.3), TIMBER, RETRO['timber'])
        link.box((-length - 0.85, yc, wz), (0.3, width + 1.7, 0.3), TIMBER, RETRO['timber'])
    p_uri, p_n = piles.write(f'{name}_piles')
    f_uri, f_n = fenders.write(f'{name}_fenders')
    link.mesh(p_uri, CONCRETE_WET, RETRO['concrete'])
    link.mesh(f_uri, TIMBER, RETRO['timber'])
    return p_n + f_n


def portal_crane(link, x, y, color):
    for dx in (-5, 5):
        for dy in (-5, 5):
            link.box((x + dx, y + dy, DECK_TOP + 7), (0.9, 0.9, 14), color, RETRO['steel'], collide=False)
    link.box((x, y, DECK_TOP + 15), (12, 12, 2), color, RETRO['steel'], collide=False)
    link.box((x, y, DECK_TOP + 19), (8, 7, 6), WHITE, RETRO['steel'], collide=False)
    link.box((x - 18, y, DECK_TOP + 30), (42, 1.6, 1.6), color, RETRO['steel'], pitch=0.55, collide=False)


def goliath_crane(link):
    y_s, y_n = DOCKS[0][1] - 12, DOCKS[1][2] + 12
    xc = 160.0
    for ly in (y_s, y_n):
        for dx in (-12, 12):
            link.box((xc + dx, ly, LAND_Z + 40), (3.0, 3.0, 80), CRANE_RED, RETRO['steel'], collide=False)
        link.box((xc, ly, LAND_Z + 1.5), (34, 4, 3), CRANE_RED, RETRO['steel'], collide=False)
    link.box((xc, (y_s + y_n) / 2, LAND_Z + 84), (30, y_n - y_s + 6, 8), CRANE_RED, RETRO['steel'], collide=False)
    link.box((xc, (y_s + y_n) / 2, LAND_Z + 84), (30.2, y_n - y_s - 20, 3), WHITE, RETRO['steel'], collide=False)
    link.box((xc, (DOCKS[0][1] + DOCKS[0][2]) / 2, LAND_Z + 76), (14, 12, 8), WHITE, RETRO['steel'], collide=False)


# --------------------------------------------------------------------------- vessels
class ShipFrame:
    """Ship frame (x fwd, y port, z up) -> yard frame."""

    def __init__(self, x, y, heading):
        self.o = np.array([x, y, 0.0])
        self.h = heading
        c, s = math.cos(heading), math.sin(heading)
        self.R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def pts(self, p):
        return np.asarray(p) @ self.R.T + self.o

    def pos(self, x, y, z):
        return tuple(self.pts([x, y, z]))


def hull_surface(L, B, T, D, bow_p=2.0, stern_p=4.0, transom=0.96, bow_rake=8.0, nx=110, nz=18):
    """Returns (below, above) grids (nx, 2*nz-1, 3) in ship frame."""
    xi = -np.cos(np.linspace(0, np.pi, nx))
    below, above = [], []
    for x in xi:
        if x >= 0:
            f = max(1 - x ** bow_p, 0.0) ** 0.5
            n = 2.0 + 4.0 * (1 - x ** 3)
        else:
            f = max(1 - (-x * transom) ** stern_p, 0.0) ** 0.5
            n = 6.0
        t_loc = T
        if x < -0.75:
            t_loc = T * (1 - 0.6 * ((-0.75 - x) / 0.25) ** 1.5)
        elif x > 0.85:
            t_loc = T * (1 - 0.5 * ((x - 0.85) / 0.15) ** 2)
        s = np.linspace(0, 1, nz)
        hb = B / 2 * f * np.clip(1 - s ** n, 0, None) ** (1 / n)
        zb = -s * t_loc
        port = np.c_[np.full(nz, x * L / 2), hb, zb]
        below.append(np.vstack([port, (port * [1, -1, 1])[::-1][1:]]))
        za = np.linspace(0, D, nz)
        fa = f + (1 - f) * 0.35 * za / D
        hba = B / 2 * fa * (1 + 0.12 * za / D)
        xa = x * L / 2 + (bow_rake * (za / D) * x ** 4 if x > 0 else -0.3 * bow_rake * (za / D) * x ** 4)
        port_a = np.c_[xa, hba, za][::-1]
        above.append(np.vstack([port_a, (port_a * [1, -1, 1])[::-1][1:]]))
    return np.array(below), np.array(above)


def build_ship_hull(name, frame, L, B, T, D, **kw):
    below, above = hull_surface(L, B, T, D, **kw)
    inside = frame.pos(0, 0, -T / 2)
    out = []
    for grid in (below, above):
        ref = grid.copy()
        ref[..., 1] = 0.0
        ref[..., 2] = grid[..., 2].mean(axis=1, keepdims=True)
        m = Mesh()
        m.grid(frame.pts(grid), ref=frame.pts(ref))
        stern = frame.pts(grid[0])
        m.fan(stern, stern.mean(axis=0), inside)
        out.append(m)
    deck = np.stack([above[:, 0], above[:, -1]], axis=1)
    out[1].grid(frame.pts(deck), ref=frame.pts(deck - [0, 0, 1.0]))
    return out


def build_carrier(link):
    c = CARRIER
    fr = ShipFrame(c['x_mid'], c['y'], math.pi)
    below, above = build_ship_hull('cvn', fr, c['L'], c['B'], c['T'], c['D'], bow_p=1.9, stern_p=5.0,
                                   transom=0.93, bow_rake=10.0)
    below.ellipsoid(fr.pos(c['L'] / 2 - 5, 0, -7.5), (11, 3.4, 3.4))
    app = Mesh()
    for py in (-16.0, -6.0, 6.0, 16.0):
        app.prism(fr.pos(-150, py, -9.0), fr.pos(-105, py * 0.8, -7.0), 0.55, sides=10)
        app.prism(fr.pos(-151.0, py, -9.0), fr.pos(-149.8, py, -9.0), 3.2, sides=20)
        app.box(fr.pos(-140, py, -5.5), (2.0, 0.5, 7.0), yaw=fr.h)
    for py in (-10.5, 10.5):
        app.box(fr.pos(-157, py, -7.0), (7.0, 0.9, 9.0), yaw=fr.h)
    for py in (-1, 1):
        for xs in np.arange(-60, 60, 2.0):
            app.box(fr.pos(xs, py * (c['B'] / 2 - 1.2), -c['T'] + 1.4), (2.05, 0.25, 1.4), yaw=fr.h)
    for m, nm, col, retro in ((below, 'cvn_hull_below', ANTIFOUL, RETRO['steel']),
                              (above, 'cvn_hull_above', GREY_HAZE, RETRO['steel']),
                              (app, 'cvn_appendages', (0.55, 0.42, 0.25), RETRO['steel'])):
        uri, _ = m.write(nm)
        link.mesh(uri, col, retro)
    h = fr.h
    link.box(fr.pos(-8, 2, c['D'] + 1.0), (300, 48, 2.0), GREY_HAZE, RETRO['steel'], yaw=h)
    link.box(fr.pos(-45, 14, c['D'] + 1.05), (225, 34, 2.0), GREY_HAZE, RETRO['steel'], yaw=h + 0.157)
    link.box(fr.pos(-45, 14, c['D'] + 2.08), (225, 0.6, 0.02), WHITE, None, yaw=h + 0.157, collide=False)
    for sx, sy in ((-60, 22), (40, -22), (-120, -20)):
        link.box(fr.pos(sx, sy, c['D'] - 3), (30, 8, 6), GREY_HAZE, RETRO['steel'], yaw=h)
    link.box(fr.pos(-30, -21, c['D'] + 2 + 9), (32, 9, 18), GREY_HAZE, RETRO['steel'], yaw=h)
    link.cylinder(fr.pos(-28, -21, c['D'] + 2 + 25), 0.9, 14, GREY_HAZE, RETRO['steel'], collide=False)
    for k in range(10):
        link.box(fr.pos(60 - k * 14, 8 + (k % 2) * 8, c['D'] + 3.2), (17, 12, 2.4), (0.5, 0.52, 0.55),
                 None, yaw=h + 0.3, collide=False)
    # Camels between hull and pier fenders.
    for xs in (-90, -20, 50, 110):
        link.cylinder(fr.pos(xs, -(c['B'] / 2 + 2.3), 0.0), 0.8, 12.0, TIMBER, RETRO['timber'],
                      pitch=math.pi / 2, yaw=h)


def build_submarine(link, name, x_mid, y, heading=math.pi):
    L, R = 115.0, 5.2
    axis_z = -4.1
    fr = ShipFrame(x_mid, y, heading)
    s = np.linspace(0, L, 90)
    r = np.full_like(s, R)
    nose = s > L - 14
    r[nose] = R * np.sqrt(np.clip(1 - ((s[nose] - (L - 14)) / 14) ** 2, 0, 1))
    tail = s < 38
    r[tail] = R * (0.25 + 0.75 * np.sin(0.5 * np.pi * s[tail] / 38) ** 1.4)
    th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    pts = np.stack([np.broadcast_to(s[:, None] - L / 2, (len(s), len(th))),
                    r[:, None] * np.cos(th), axis_z + r[:, None] * np.sin(th)], -1)
    axis_pts = np.stack([pts[..., 0], np.zeros_like(pts[..., 0]), np.full_like(pts[..., 0], axis_z)], -1)
    m = Mesh()
    m.grid(fr.pts(pts.transpose(1, 0, 2)), wrap_u=True, ref=fr.pts(axis_pts.transpose(1, 0, 2)))
    m.fan(fr.pts(pts[0]), fr.pos(-L / 2, 0, axis_z), fr.pos(0, 0, axis_z))
    m.box(fr.pos(L / 2 - 30, 0, axis_z + R + 3.0), (9.0, 2.6, 7.0), yaw=heading)
    m.ellipsoid(fr.pos(L / 2 - 25.5, 0, axis_z + R + 3.0), (1.6, 1.3, 3.5), n=8)
    for ang in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        cy, cz = math.cos(ang), math.sin(ang)
        m.box(fr.pos(-L / 2 + 7.0, cy * (R * 0.35 + 2.0), axis_z + cz * (R * 0.35 + 2.0)),
              (5.0, 0.4 + 3.6 * abs(cy), 0.4 + 3.6 * abs(cz)), yaw=heading)
    # Pump-jet shroud: outer skin faces away from the shaft axis, inner skin towards it.
    th2 = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    xs = np.array([-L / 2 - 2.5, -L / 2 + 1.0])
    for rad, ref_rad in ((1.9, 0.0), (1.7, 10.0)):
        ring = np.stack([np.broadcast_to(xs[:, None], (2, 28)), rad * np.cos(th2)[None].repeat(2, 0),
                         axis_z + rad * np.sin(th2)[None].repeat(2, 0)], -1)
        ref = ring.copy()
        ref[..., 1] *= ref_rad / rad
        ref[..., 2] = axis_z + (ring[..., 2] - axis_z) * ref_rad / rad
        m.grid(fr.pts(ring.transpose(1, 0, 2)), wrap_u=True, ref=fr.pts(ref.transpose(1, 0, 2)))
    uri, _ = m.write(name)
    link.mesh(uri, SUB_BLACK, RETRO['anechoic'], specular=0.05)
    for xs in (-25.0, 25.0):
        side = -1 if heading == math.pi else 1
        link.cylinder(fr.pos(xs, side * -(R + 1.4), 0.0), 0.7, 10.0, (0.1, 0.1, 0.1), RETRO['rubber'],
                      pitch=math.pi / 2, yaw=heading)


def build_workboats(link):
    fr = ShipFrame(-150.0, 330.0 - 15.0 - 1.5 - 9.0, math.pi)
    link.box(fr.pos(0, 0, 0.5), (60.0, 18.0, 5.0), STEEL_RUST, RETRO['steel'], yaw=fr.h)
    link.box(fr.pos(-15, 0, 4.0), (10, 8, 2), CRANE_YELLOW, RETRO['steel'], yaw=fr.h, collide=False)
    link.box(fr.pos(5, 0, 14.0), (32, 1.4, 1.4), CRANE_YELLOW, RETRO['steel'], yaw=fr.h, pitch=-0.6,
             collide=False)
    for k, (x, y, h) in enumerate([(-260.0, -95.0, math.pi * 0.95), (-310.0, 160.0, math.pi * 1.1)]):
        tug = ShipFrame(x, y, h)
        below, above = build_ship_hull('tug', tug, 28.0, 10.0, 4.2, 2.5, bow_p=1.6, stern_p=3.0,
                                       transom=0.9, bow_rake=2.0, nx=40, nz=10)
        u1, _ = below.write(f'tug_{k}_below')
        u2, _ = above.write(f'tug_{k}_above')
        link.mesh(u1, ANTIFOUL, RETRO['steel'])
        link.mesh(u2, (0.15, 0.2, 0.35), RETRO['steel'])
        link.box(tug.pos(2, 0, 4.5), (10, 7, 4), WHITE, RETRO['steel'], yaw=h, collide=False)
        link.box(tug.pos(4, 0, 8.0), (5, 5.5, 3), WHITE, RETRO['steel'], yaw=h, collide=False)
        link.cylinder(tug.pos(-9.5, 0, -2.6), 1.1, 0.4, (0.6, 0.45, 0.2), RETRO['steel'],
                      pitch=math.pi / 2, yaw=h)


# --------------------------------------------------------------------------- seabed clutter
def water_xy(x_range=(-320.0, -4.0), y_range=(-680.0, 680.0)):
    while True:
        x, y = RNG.uniform(*x_range), RNG.uniform(*y_range)
        under_pier = any(-length - 2 < x < 0 and abs(y - yc) < width / 2 + 1 for _, yc, length, width in PIERS)
        if not under_pier or RNG.random() < 0.25:
            return x, y


def build_clutter(link):
    rocks = Mesh()
    gaps = dock_gaps()
    for y in np.arange(SHORE_Y[0] + 2, SHORE_Y[1] - 2, 1.3):
        if any(g0 - 2 < y < g1 + 2 for g0, g1 in gaps):
            continue
        for _ in range(2):
            x = RNG.uniform(-7.0, -0.8)
            size = RNG.uniform(0.4, 1.3)
            rocks.rock((x, y + RNG.uniform(-0.6, 0.6), ground_z(x, y) + 0.25 * size), size)
    uri, _ = rocks.write('riprap')
    link.mesh(uri, (0.38, 0.36, 0.33), RETRO['rock'])

    junk = Mesh()
    for _ in range(40):
        x, y = water_xy()
        junk.torus((x, y, ground_z(x, y) + 0.12), 0.42, 0.14, normal_axis=RNG.choice(['z', 'x']))
    for _ in range(25):
        x, y = water_xy()
        junk.prism((x, y, ground_z(x, y) + 0.3), (x + 0.01, y, ground_z(x, y) + 0.9), 0.29, sides=12)
    for _ in range(15):
        x, y = water_xy()
        a = RNG.uniform(0, np.pi)
        ln = RNG.uniform(3, 9)
        rad = RNG.uniform(0.3, 0.6)
        z = ground_z(x, y) + rad * 0.8
        junk.prism((x, y, z), (x + ln * math.cos(a), y + ln * math.sin(a), z), rad, sides=14, cap=False)
    for _ in range(18):
        x, y = water_xy()
        s = RNG.uniform(0.8, 2.2, 3)
        junk.box((x, y, ground_z(x, y) + s[2] / 2 - 0.2), s, yaw=RNG.uniform(0, np.pi))
    uri, _ = junk.write('debris')
    link.mesh(uri, (0.30, 0.28, 0.25), RETRO['debris'])

    chain = Mesh()
    for bx, by in ((-460.0, -300.0), (-470.0, 0.0), (-455.0, 300.0)):
        gz_ = ground_z(bx, by)
        link.cylinder((bx, by, 0.3), 1.2, 1.8, (0.8, 0.15, 0.1), RETRO['steel'])
        link.box((bx + 6, by, gz_ + 0.4), (2.0, 2.0, 1.2), CONCRETE_WET, RETRO['concrete'])
        n = 60
        for k in range(n):
            t = k / n
            px = bx + 6 * t
            pz = -0.6 + (gz_ + 1.0 + 0.6) * t + 2.5 * math.sin(math.pi * t) * -0.5
            q = px + 0.18
            chain.prism((px, by, pz), (q, by, pz + (gz_ + 1.6) / n), 0.06, sides=5)
    x, y = -205.0, 112.0
    gzz = ground_z(x, y)
    chain.box((x, y, gzz + 0.15), (0.4, 3.0, 0.3))
    chain.box((x, y, gzz + 1.2), (0.4, 0.4, 2.2))
    for k in range(40):
        chain.prism((x + 0.15 * k, y + 0.05 * k * math.sin(k), gzz + 0.1), (x + 0.15 * (k + 1), y, gzz + 0.1),
                    0.05, sides=5)
    uri, _ = chain.write('chains')
    link.mesh(uri, STEEL_RUST, RETRO['steel'])

    # Stubs of a demolished timber pier at the south end of the yard.
    stubs = Mesh()
    for sx in np.arange(-150.0, -15.0, 6.0):
        for sy in np.arange(-655.0, -625.0, 4.0):
            if RNG.random() < 0.3:
                continue
            gz_ = ground_z(sx, sy)
            stubs.prism((sx, sy, gz_ - 0.5), (sx, sy, gz_ + RNG.uniform(0.3, 2.5)), RNG.uniform(0.15, 0.22),
                        sides=7)
    uri, _ = stubs.write('pile_stubs')
    link.mesh(uri, TIMBER, RETRO['timber'])

    # Sunken skiff.
    skiff = ShipFrame(-95.0, 60.0, 0.7)
    below, _ = build_ship_hull('skiff', skiff, 7.0, 2.4, 0.9, 0.2, bow_p=1.5, stern_p=3, transom=0.9,
                               bow_rake=0.5, nx=16, nz=6)
    gzz = ground_z(-95.0, 60.0)
    below.v = [(a, b, c * -1 + gzz + 0.1) for a, b, c in below.v]
    below.f = [(a, c, b) for a, b, c in below.f]
    uri, _ = below.write('skiff')
    link.mesh(uri, (0.6, 0.6, 0.55), RETRO['debris'])

    for y in np.arange(SHORE_Y[0] + 20, SHORE_Y[1], 40.0):
        if any(g0 - 10 < y < g1 + 10 for g0, g1 in gaps) or any(abs(y - yc) < w / 2 + 6 for _, yc, _, w in PIERS):
            continue
        link.cylinder((-1.4, y, 0.2), 1.0, 3.2, (0.08, 0.08, 0.08), RETRO['rubber'], roll=math.pi / 2)


# --------------------------------------------------------------------------- land
def build_buildings(link):
    occupied = [(0, LAND_DEPTH_X, y0 - 15, y1 + 15) for _, y0, y1, _, _ in DOCKS]
    occupied.append((140, 185, DOCKS[0][1] - 20, DOCKS[1][2] + 20))
    roofs = [(0.55, 0.62, 0.70), (0.70, 0.70, 0.68), (0.45, 0.50, 0.58), (0.62, 0.58, 0.52)]
    placed = 0
    attempts = 0
    while placed < 38 and attempts < 2000:
        attempts += 1
        sx, sy = RNG.uniform(40, 140), RNG.uniform(30, 160)
        cx, cy = RNG.uniform(35 + sx / 2, LAND_DEPTH_X - sx / 2), RNG.uniform(SHORE_Y[0] + sy / 2, SHORE_Y[1] - sy / 2)
        box = (cx - sx / 2 - 8, cx + sx / 2 + 8, cy - sy / 2 - 8, cy + sy / 2 + 8)
        if any(not (box[1] < o[0] or box[0] > o[1] or box[3] < o[2] or box[2] > o[3]) for o in occupied):
            continue
        occupied.append(box)
        h = RNG.uniform(10, 38)
        link.box((cx, cy, LAND_Z + h / 2), (sx, sy, h), roofs[placed % len(roofs)], None, collide=False)
        placed += 1


# --------------------------------------------------------------------------- world
def yard_to_world(x, y):
    c, s = math.cos(YARD_YAW), math.sin(YARD_YAW)
    return c * x - s * y, s * x + c * y


def world_xml(models, rov_pose, leader_pose):
    return f'''<?xml version="1.0"?>
<!-- Generated by scripts/generate_nns_world.py; edit the generator, not this file. -->
<sdf version="1.9">
  <world name="nns_shipyard">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="libgz-sim-imu-system.so" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-buoyancy-system" name="gz::sim::systems::Buoyancy">
      <graded_buoyancy>
        <default_density>1000</default_density>
        <density_change>
          <above_depth>0.0</above_depth>
          <density>1</density>
        </density_change>
      </graded_buoyancy>
      <enable>bluerov2_heavy</enable>
    </plugin>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{ORIGIN_LAT}</latitude_deg>
      <longitude_deg>{ORIGIN_LON}</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>

    <scene>
      <ambient>0.35 0.40 0.40 1</ambient>
      <background>0.42 0.55 0.62 1</background>
      <shadows>false</shadows>
      <grid>false</grid>
    </scene>

    <!-- Shadows off: over a 1.4 km scene the shadow map aliases badly on thin piles. -->
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 100 0 0 0</pose>
      <diffuse>0.9 0.9 0.85 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>-0.4 0.3 -0.85</direction>
    </light>

    <model name="water_surface">
      <static>true</static>
      <link name="link">
        <!-- Downward-facing plane keeps the positively buoyant ROV from leaving the water. -->
        <collision name="ceiling">
          <pose>0 0 0.1 3.14159 0 0</pose>
          <geometry><plane><normal>0 0 1</normal><size>4000 4000</size></plane></geometry>
        </collision>
        <visual name="surface">
          <cast_shadows>false</cast_shadows>
          <transparency>0.35</transparency>
          <geometry><plane><normal>0 0 1</normal><size>4000 4000</size></plane></geometry>
          {_material((0.14, 0.27, 0.27), specular=0.6, alpha=0.65, double_sided=True)}
        </visual>
      </link>
    </model>

    {models}

    <include>
      <uri>model://bluerov2_heavy</uri>
      {_pose(*rov_pose)}
    </include>

    <include>
      <uri>model://bluerov2_dummy</uri>
      <name>bluerov2_leader</name>
      {_pose(*leader_pose)}
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">
        <label>255</label>
      </plugin>
    </include>
  </world>
</sdf>
'''


def main():
    os.makedirs(MESH_DIR, exist_ok=True)
    for f in os.listdir(MESH_DIR):
        os.remove(os.path.join(MESH_DIR, f))
    tex_dir = os.path.join(MODEL_DIR, 'materials', 'textures')
    os.makedirs(tex_dir, exist_ok=True)
    shutil.copy(os.path.join(PKG, 'models', 'sand_heightmap', 'materials', 'soil_sand_0045_01.jpg'),
                os.path.join(tex_dir, 'seabed.jpg'))

    seabed = Link('seabed')
    uri, n_seabed = build_seabed()
    seabed.mesh(uri, (0.50, 0.46, 0.38), RETRO['mud'], specular=0.02,
                albedo_map='model://nns_shipyard/materials/textures/seabed.jpg')

    bulkhead = Link('bulkhead')
    land = Link('land')
    docks = Link('dry_docks')
    (uri, _), segments = build_bulkhead()
    bulkhead.mesh(uri, (0.30, 0.25, 0.20), RETRO['steel'])
    build_land_and_docks(land, docks, bulkhead, segments)

    piers = Link('piers')
    tris = sum(build_pier(name, yc, ln, w, piers) for name, yc, ln, w in PIERS)
    for name, yc, ln, w in PIERS[::2]:
        portal_crane(piers, -ln * 0.3, yc, CRANE_YELLOW)
        portal_crane(piers, -ln * 0.7, yc, CRANE_RED)

    cranes = Link('cranes')
    goliath_crane(cranes)
    buildings = Link('buildings')
    build_buildings(buildings)
    clutter = Link('seabed_clutter')
    build_clutter(clutter)

    vessels = Link('vessels')
    build_carrier(vessels)
    for sub in SUBS:
        build_submarine(vessels, sub['name'], sub['x_mid'], sub['y'])
    build_workboats(vessels)

    models = '\n    '.join([
        model_xml('nns_seabed', [seabed], 200),
        model_xml('nns_shipyard', [bulkhead, land, docks, piers, cranes, buildings, clutter], 100),
        model_xml('nns_vessels', [vessels], 110),
    ])

    rx, ry, rz, ryaw = ROV_SPAWN_YARD
    wx, wy = yard_to_world(rx, ry)
    lx, ly = yard_to_world(rx + 6.0 * math.cos(ryaw), ry + 6.0 * math.sin(ryaw))
    rov = (wx, wy, rz, 0, 0, ryaw + YARD_YAW)
    leader = (lx, ly, rz - 1.0, 0, 0, ryaw + YARD_YAW)
    with open(WORLD_PATH, 'w') as fh:
        fh.write(world_xml(models, rov, leader))

    with open(os.path.join(MODEL_DIR, 'model.config'), 'w') as fh:
        fh.write('<?xml version="1.0"?>\n<model>\n  <name>nns_shipyard</name>\n  <version>1.0</version>\n'
                 '  <description>Mesh assets for worlds/nns_shipyard.world (generated).</description>\n'
                 '</model>\n')
    print(f'seabed tris: {n_seabed}, pier tris: {tris}')
    print(f'world: {WORLD_PATH}')
    print(f'ROV spawn (world): {rov[0]:.2f} {rov[1]:.2f} {rov[2]:.2f} yaw {rov[5]:.3f}')


if __name__ == '__main__':
    main()
