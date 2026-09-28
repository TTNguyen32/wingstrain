/* Wing strain -> downforce: the forward model, ported from the Python package.
 *
 * Scope of this interactive version: two symmetric mounts, symmetric loading.
 * That makes the beam statically determinate, so the internal loads come from
 * statics alone and no linear solver is needed. The full package handles
 * arbitrary mounts, taper and the multi-gauge MAP inverse.
 *
 * Car axes: x aft, y spanwise, z up. Downforce is -z.
 * M_x > 0 means tension on the top skin.
 */

// ---------------------------------------------------------------- materials
export const PLIES = {
  twill_200: { name: "200 gsm 2x2 twill", E1: 61e9, E2: 61e9, G12: 4.2e9, nu12: 0.05, t: 0.25e-3, rho: 1500 },
  twill_300: { name: "300 gsm 2x2 twill", E1: 60e9, E2: 60e9, G12: 4.2e9, nu12: 0.05, t: 0.36e-3, rho: 1500 },
  ud_t700:   { name: "UD T700 150 gsm",   E1: 135e9, E2: 9e9, G12: 4.8e9, nu12: 0.28, t: 0.15e-3, rho: 1570 },
};

function qMatrix(p) {
  const nu21 = p.nu12 * p.E2 / p.E1, d = 1 - p.nu12 * nu21;
  return [p.E1 / d, p.nu12 * p.E2 / d, p.E2 / d, p.G12]; // Q11,Q12,Q22,Q66
}

function qBar(Q, deg) {
  const t = deg * Math.PI / 180, m = Math.cos(t), n = Math.sin(t);
  const [Q11, Q12, Q22, Q66] = Q;
  const m2 = m * m, n2 = n * n, m4 = m2 * m2, n4 = n2 * n2;
  return {
    11: Q11 * m4 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * n4,
    12: (Q11 + Q22 - 4 * Q66) * m2 * n2 + Q12 * (m4 + n4),
    22: Q11 * n4 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * m4,
    16: (Q11 - Q12 - 2 * Q66) * m2 * m * n + (Q12 - Q22 + 2 * Q66) * m * n2 * n,
    26: (Q11 - Q12 - 2 * Q66) * m * n2 * n + (Q12 - Q22 + 2 * Q66) * m2 * m * n,
    66: (Q11 + Q22 - 2 * Q12 - 2 * Q66) * m2 * n2 + Q66 * (m4 + n4),
  };
}

function inv3(A) {
  const [a, b, c, d, e, f, g, h, i] = A;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  return [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
          (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
          (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
}

/** Classical laminate theory: ply angles (deg from spanwise) -> membrane moduli. */
export function laminate(angles, plyKey = "twill_200") {
  const p = PLIES[plyKey], Q = qMatrix(p);
  const A = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (const ang of angles) {
    const q = qBar(Q, ang);
    A[0] += q[11] * p.t;  A[1] += q[12] * p.t;  A[2] += q[16] * p.t;
    A[3] += q[12] * p.t;  A[4] += q[22] * p.t;  A[5] += q[26] * p.t;
    A[6] += q[16] * p.t;  A[7] += q[26] * p.t;  A[8] += q[66] * p.t;
  }
  const h = angles.length * p.t, a = inv3(A);
  return {
    thickness: h, Ex: 1 / (a[0] * h), Ey: 1 / (a[4] * h), Gxy: 1 / (a[8] * h),
    nu_xy: -a[1] / a[0], arealMass: h * p.rho, plies: angles.length, ply: p,
  };
}

// ----------------------------------------------------------------- aerofoil
/** Sorted, de-duplicated surface -> a linear interpolator in x. */
function surfaceInterp(X, Z) {
  const o = X.map((v, i) => i).sort((a, b) => X[a] - X[b]);
  const x = [], z = [];
  for (const i of o) {
    if (x.length && Math.abs(X[i] - x[x.length - 1]) < 1e-12) continue;
    x.push(X[i]); z.push(Z[i]);
  }
  return (q) => {
    if (q <= x[0]) return z[0];
    if (q >= x[x.length - 1]) return z[z.length - 1];
    let lo = 0, hi = x.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (x[m] <= q) lo = m; else hi = m; }
    const f = (q - x[lo]) / (x[hi] - x[lo] || 1);
    return z[lo] + f * (z[hi] - z[lo]);
  };
}

function makeProfile(name, upX, upZ, loX, loZ, nPts) {
  const zUp = surfaceInterp(upX, upZ), zLo = surfaceInterp(loX, loZ);
  let tmax = 0, xtmax = 0, camber = 0;
  for (let i = 0; i <= 200; i++) {
    const x = i / 200, t = zUp(x) - zLo(x);
    if (t > tmax) { tmax = t; xtmax = x; }
    const c = (zUp(x) + zLo(x)) / 2;
    if (Math.abs(c) > Math.abs(camber)) camber = c;
  }
  return { name, zUp, zLo, tmax, xtmax, camber, nPts: nPts ?? upX.length + loX.length };
}

/** NACA 4-digit surfaces, cosine spaced. */
export function naca4(code, n = 140) {
  const m = +code[0] / 100, pp = +code[1] / 10, th = +code.slice(2) / 100;
  const xu = [], zu = [], xl = [], zl = [];
  for (let i = 0; i < n; i++) {
    const x = (1 - Math.cos(Math.PI * i / (n - 1))) / 2;
    const yt = 5 * th * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x
                         + 0.2843 * x ** 3 - 0.1036 * x ** 4);
    let yc = 0, dyc = 0;
    if (m > 0 && pp > 0) {
      if (x < pp) { yc = m / (pp * pp) * (2 * pp * x - x * x); dyc = 2 * m / (pp * pp) * (pp - x); }
      else { yc = m / (1 - pp) ** 2 * ((1 - 2 * pp) + 2 * pp * x - x * x); dyc = 2 * m / (1 - pp) ** 2 * (pp - x); }
    }
    const a = Math.atan(dyc);
    xu.push(x - yt * Math.sin(a)); zu.push(yc + yt * Math.cos(a));
    xl.push(x + yt * Math.sin(a)); zl.push(yc - yt * Math.cos(a));
  }
  return makeProfile(`NACA ${code}`, xu, zu, xl, zl);
}

/** Parse a coordinate file: Selig or Lednicer order, any units, any datum.
 *  Throws an Error with a plain-language reason if the contour is not usable. */
export function parseProfile(text, fallbackName = "custom") {
  const pts = [];
  let header = null;
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || /^[#!;]/.test(line)) continue;
    const parts = line.replace(/,/g, " ").split(/\s+/);
    const a = Number(parts[0]), b = Number(parts[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) { header = header ?? line; continue; }  // name as written
    pts.push([a, b]);
  }
  if (!pts.length) throw new Error("No coordinate pairs found in that file.");
  // Lednicer point-count header: two whole numbers that add up to what follows
  const h = pts[0];
  if (pts.length > 1 && h.every(v => Math.abs(v - Math.round(v)) < 1e-6 && v >= 3)
      && Math.abs(h[0] + h[1] - (pts.length - 1)) <= 2) pts.shift();
  if (pts.length < 10)
    throw new Error(`Only ${pts.length} points — an aerofoil needs at least 10.`);

  const xs = pts.map(p => p[0]);
  const x0 = Math.min(...xs), chord = Math.max(...xs) - x0;
  if (!(chord > 0))
    throw new Error("Every point shares one x position — that is not an aerofoil.");
  // datum on the leading edge, so a profile exported from a CAD sketch
  // somewhere off the origin does not pick up phantom camber
  const z0 = pts[xs.indexOf(x0)][1];
  const P = pts.map(([x, z]) => [(x - x0) / chord, (z - z0) / chord]);

  let runs;
  if (P[0][0] > 0.5) {                      // Selig: starts at the trailing edge
    let i = 0;
    P.forEach((p, k) => { if (p[0] < P[i][0]) i = k; });
    runs = [P.slice(0, i + 1).reverse(), P.slice(i)];
  } else {                                  // two runs, each from the nose aft
    let i = 1, worst = Infinity;
    for (let k = 1; k < P.length; k++) {
      const d = P[k][0] - P[k - 1][0];
      if (d < worst) { worst = d; i = k; }
    }
    if (i < 4 || P.length - i < 4) {
      let j = 0;
      P.forEach((p, k) => { if (p[0] > P[j][0]) j = k; });
      i = j + 1;
    }
    runs = [P.slice(0, i), P.slice(i)];
  }
  runs = runs.filter(r => r.length >= 4);
  if (runs.length !== 2)
    throw new Error("Could not split the contour into an upper and a lower surface.");
  const share = Math.min(...runs.map(r => Math.max(...r.map(p => p[0]))))
              - Math.max(...runs.map(r => Math.min(...r.map(p => p[0]))));
  if (share < 0.5)
    throw new Error(`The two surfaces only share ${Math.max(0, share * 100).toFixed(0)}% `
      + "of the chord — this looks like a single open curve, not a closed contour.");
  const mean = (r) => r.reduce((a, p) => a + p[1], 0) / r.length;
  if (mean(runs[0]) < mean(runs[1])) runs.reverse();
  const prof = makeProfile(header || fallbackName,
    runs[0].map(p => p[0]), runs[0].map(p => p[1]),
    runs[1].map(p => p[0]), runs[1].map(p => p[1]), pts.length);
  if (!(prof.tmax > 0)) throw new Error("The two surfaces cross — check the point order.");
  return prof;
}

/** Aerofoil axes -> car axes: scale, invert for a downforce wing, rotate. */
export function toCar(xc, zc, chord, angleDeg, inverted, xRef) {
  let x = xc * chord, z = zc * chord;
  if (inverted) z = -z;
  const b = angleDeg * Math.PI / 180, cb = Math.cos(b), sb = Math.sin(b);
  x -= xRef * chord;
  return [x * cb + z * sb, -x * sb + z * cb];
}

// ------------------------------------------------------------------ section
/** Single closed cell: skin from x_front to x_rear plus a rear shear web. */
export function buildSection(af, P) {
  const { chord, angle, inverted, xRef, xFront, xRear, skin, web, nSkin } = P;
  const cos = (a, b, n) => Array.from({ length: n },
    (_, i) => a + (b - a) * (1 - Math.cos(Math.PI * i / (n - 1))) / 2);
  const xs = cos(xFront, xRear, nSkin);
  const runs = [];
  // physical labels: for an inverted wing the aerofoil's upper surface faces down
  const top = inverted ? "lower" : "upper", bot = inverted ? "upper" : "lower";
  const zf = (s) => (s === "upper" ? af.zUp : af.zLo);
  runs.push({ pts: xs.slice().reverse().map(x => [x, af.zLo(x)]),
              xcs: xs.slice().reverse(), lam: skin, surf: inverted ? "top" : "bottom" });
  runs.push({ pts: xs.map(x => [x, af.zUp(x)]), xcs: xs,
              lam: skin, surf: inverted ? "bottom" : "top" });
  if (xRear < 0.999) {
    runs.push({ pts: [[xRear, af.zUp(xRear)], [xRear, af.zLo(xRear)]],
                xcs: [xRear, xRear], lam: web, surf: "web_rear" });
  }
  let segs = [];
  for (const r of runs) {
    const P2 = r.pts.map(([a, b]) => toCar(a, b, chord, angle, inverted, xRef));
    for (let k = 0; k < P2.length - 1; k++) {
      const p0 = P2[k], p1 = P2[k + 1];
      const ds = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
      if (ds < 1e-9) continue;
      segs.push({ p0, p1, ds, mid: [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2],
                  lam: r.lam, surf: r.surf,
                  xc: (r.xcs[k] + r.xcs[k + 1]) / 2 });
    }
  }
  // close the loop exactly, then force counter-clockwise
  for (let i = 0; i < segs.length - 1; i++) segs[i + 1].p0 = segs[i].p1;
  segs[segs.length - 1].p1 = segs[0].p0;
  const area = (S) => 0.5 * S.reduce((a, s) =>
    a + s.p0[0] * s.p1[1] - s.p1[0] * s.p0[1], 0);
  if (area(segs) < 0) {
    segs = segs.reverse().map(s => ({ ...s, p0: s.p1, p1: s.p0 }));
  }
  for (const s of segs) {
    s.ds = Math.hypot(s.p1[0] - s.p0[0], s.p1[1] - s.p0[1]);
    s.mid = [(s.p0[0] + s.p1[0]) / 2, (s.p0[1] + s.p1[1]) / 2];
  }

  // E-weighted section properties
  let EA = 0, Sx = 0, Sz = 0, mass = 0;
  for (const s of segs) {
    const w = s.lam.Ex * s.lam.thickness * s.ds;
    EA += w; Sx += w * s.mid[1]; Sz += w * s.mid[0];
    mass += s.lam.arealMass * s.ds;
  }
  const xEA = Sz / EA, zEA = Sx / EA;
  let EIxx = 0, EIzz = 0, EIxz = 0;
  for (const s of segs) {
    const w = s.lam.Ex * s.lam.thickness * s.ds;
    const dx = s.mid[0] - xEA, dz = s.mid[1] - zEA;
    EIxx += w * dz * dz; EIzz += w * dx * dx; EIxz += w * dx * dz;
    const tx = (s.p1[0] - s.p0[0]) / s.ds, tz = (s.p1[1] - s.p0[1]) / s.ds;
    EIxx += s.lam.Ex * s.ds ** 3 * s.lam.thickness / 12 * tz * tz;
    EIzz += s.lam.Ex * s.ds ** 3 * s.lam.thickness / 12 * tx * tx;
  }
  const Aenc = area(segs);
  const circ = segs.reduce((a, s) => a + s.ds / (s.lam.Gxy * s.lam.thickness), 0);
  const GJ = 4 * Aenc * Aenc / circ;

  // shear-flow analysis: q per unit dMx/dy, and the shear centre
  let acc = 0;
  const qb = segs.map(s => {
    acc += s.lam.Ex * s.lam.thickness * s.ds * (s.mid[1] - zEA);
    return -acc / EIxx;
  });
  let num = 0, den = 0;
  segs.forEach((s, i) => {
    const f = s.ds / (s.lam.Gxy * s.lam.thickness);
    num += qb[i] * f; den += f;
  });
  const qhat = qb.map(v => v - num / den);
  let Fz = 0, My = 0;
  segs.forEach((s, i) => {
    const dx = s.p1[0] - s.p0[0], dz = s.p1[1] - s.p0[1];
    Fz += qhat[i] * dz;
    My += qhat[i] * ((s.mid[0] - xEA) * dz - (s.mid[1] - zEA) * dx);
  });
  const xSC = xEA + (Math.abs(Fz) > 1e-12 ? My / Fz : 0);

  const zs = segs.map(s => s.mid[1]);
  return { segs, qhat, EA, xEA, zEA, EIxx, EIzz, EIxz, Aenc, GJ, xSC, mass,
           zMax: Math.max(...zs), zMin: Math.min(...zs),
           depth: Math.max(...zs) - Math.min(...zs) };
}

/** Strain plane for coupled (unsymmetric) bending. */
export function epsYY(sec, x, z, Mx, Mz) {
  const det = sec.EIzz * sec.EIxx - sec.EIxz * sec.EIxz;
  const b = (Mz * sec.EIxx - Mx * sec.EIxz) / det;
  const c = (Mx * sec.EIzz - Mz * sec.EIxz) / det;
  return b * (x - sec.xEA) + c * (z - sec.zEA);
}

export function locate(sec, xc, surface) {
  let best = -1, bd = Infinity;
  sec.segs.forEach((s, i) => {
    if (s.surf !== surface) return;
    const d = Math.abs(s.xc - xc);
    if (d < bd) { bd = d; best = i; }
  });
  return best;
}

// --------------------------------------------------------------------- load
export function spanShape(kind, tipLoading, span, n = 241) {
  const y = [], f = [];
  for (let i = 0; i < n; i++) {
    const yy = span * i / (n - 1), s = 2 * i / (n - 1) - 1;
    y.push(yy);
    f.push(kind === "uniform" ? 1
      : kind === "elliptical" ? Math.sqrt(Math.max(0, 1 - s * s))
      : 1 - (1 - tipLoading) * s * s);
  }
  let I = 0;
  for (let i = 0; i < n - 1; i++) I += 0.5 * (f[i] + f[i + 1]) * (y[i + 1] - y[i]);
  for (let i = 0; i < n; i++) f[i] /= I;
  return (q) => {
    if (q <= 0) return f[0];
    if (q >= span) return f[n - 1];
    const t = q / span * (n - 1), i = Math.min(n - 2, Math.floor(t));
    return f[i] + (t - i) * (f[i + 1] - f[i]);
  };
}

/** Statics for two symmetric mounts: shear, bending, torque along the span. */
export function beamStatics(P, sec, loads, nBase = 240) {
  const { span, mountY } = P;
  const y1 = mountY, y2 = span - mountY;
  const pts = loads.pointLoads;               // [{y, Fz, torque}]
  // stations: a uniform grid plus the exact mount and bracket positions, so
  // concentrated loads land on a node instead of being nudged to the nearest one
  const want = [y1, y2, ...pts.map(p => p.y)];
  const y = [...new Set([...Array.from({ length: nBase + 1 },
    (_, i) => +(span * i / nBase).toFixed(9)),
    ...want.map(v => +v.toFixed(9))])]
    .filter(v => v >= 0 && v <= span).sort((a, b) => a - b);
  const nSta = y.length;

  // reactions: symmetric layout, so each mount takes half the total
  const total = loads.Dtotal;                 // downforce, positive
  const R = total / 2;

  const jump = new Array(nSta).fill(0);
  const idx = (yy) => {
    let best = 0, bd = Infinity;
    for (let i = 0; i < nSta; i++) {
      const d = Math.abs(y[i] - yy);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  };
  for (const p of pts) jump[idx(p.y)] += p.Fz;
  jump[idx(y1)] += R; jump[idx(y2)] += R;

  const Sh = new Array(nSta).fill(0), Mx = new Array(nSta).fill(0);
  Sh[0] = jump[0];
  for (let i = 0; i < nSta - 1; i++) {
    const h = y[i + 1] - y[i], gm = h / (2 * Math.sqrt(3));
    const ym = (y[i] + y[i + 1]) / 2;
    const w = 0.5 * (loads.pz(ym - gm) + loads.pz(ym + gm));
    const sBefore = Sh[i] + w * h;
    Mx[i + 1] = Mx[i] - 0.5 * (Sh[i] + sBefore) * h;
    Sh[i + 1] = sBefore + jump[i + 1];
  }

  // chordwise bending from drag, same construction
  const Sx = new Array(nSta).fill(0), Mz = new Array(nSta).fill(0);
  const jx = new Array(nSta).fill(0);
  for (const p of pts) jx[idx(p.y)] += p.Fx || 0;
  const Rx = -loads.dragTotal / 2;
  jx[idx(y1)] += Rx; jx[idx(y2)] += Rx;
  Sx[0] = jx[0];
  for (let i = 0; i < nSta - 1; i++) {
    const h = y[i + 1] - y[i], ym = (y[i] + y[i + 1]) / 2;
    const w = loads.px(ym);
    const sBefore = Sx[i] + w * h;
    Mz[i + 1] = Mz[i] - 0.5 * (Sx[i] + sBefore) * h;
    Sx[i + 1] = sBefore + jx[i + 1];
  }

  // torque: symmetric loading, so each mount reacts half the applied torque
  const tq = (yy) => -(loads.xcp(yy) - sec.xSC) * loads.pz(yy)
                     + (loads.zcp(yy) - sec.zEA) * loads.px(yy);
  const acc = new Array(nSta).fill(0);
  for (let i = 1; i < nSta; i++) {
    const h = y[i] - y[i - 1], ym = (y[i] + y[i - 1]) / 2;
    acc[i] = acc[i - 1] + tq(ym) * h;
  }
  let Ttot = acc[nSta - 1];
  for (const p of pts) Ttot += p.torque || 0;
  const tj = new Array(nSta).fill(0);
  for (const p of pts) tj[idx(p.y)] += p.torque || 0;
  tj[idx(y1)] -= Ttot / 2; tj[idx(y2)] -= Ttot / 2;
  const T = new Array(nSta).fill(0);
  let run = 0;
  for (let i = 0; i < nSta; i++) {
    run += tj[i];
    T[i] = -(acc[i] + run);
  }
  return { y, Mx, Mz, T, Sz: Sh, reaction: R };
}

const at = (y, arr, q) => {
  const n = y.length;
  if (q <= y[0]) return arr[0];
  if (q >= y[n - 1]) return arr[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (y[m] <= q) lo = m; else hi = m; }
  const f = (q - y[lo]) / (y[hi] - y[lo]);
  return arr[lo] + f * (arr[hi] - arr[lo]);
};
export { at };

// ------------------------------------------------------------------ gauges
export function resolveStrain(eSpan, eChord, gamma, angleDeg) {
  const t = angleDeg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  return eSpan * c * c + eChord * s * s + gamma * s * c;
}

export const BRIDGE = { quarter: 0.25, half: 0.5, full: 1.0, full_poisson: 0.65 };

// --------------------------------------------------------------------- solve
export function solve(P) {
  const af = P.profile || naca4(P.naca);
  const skin = laminate(P.skinAngles, P.plyKey);
  const web = laminate([45, -45], P.plyKey);
  const sec = buildSection(af, { ...P, skin, web, nSkin: P.nSkin || 72 });

  const q = 0.5 * P.rho * P.V * P.V;
  const area = P.span * P.chord;
  const Del = q * area * P.CL;
  const Dcar = q * P.CLS_flaps;
  const Dtot = Del + Dcar;
  const dragEl = q * area * P.CD;
  const shape = spanShape(P.dist, P.tipLoading, P.span);
  const cp = (xc) => toCar(xc, 0, P.chord, P.angle, P.inverted, P.xRef);

  const loads = {
    Dtotal: Dtot, dragTotal: dragEl + q * P.CDS_flaps,
    pz: (y) => -Del * shape(y),
    px: (y) => dragEl * shape(y),
    xcp: () => cp(P.xcCP)[0], zcp: () => cp(P.xcCP)[1],
    pointLoads: [],
  };
  if (Dcar > 0) {
    const [xa, za] = cp(P.flapXc);
    for (const fr of [P.bracket, 1 - P.bracket]) {
      const Fz = -Dcar / 2, Fx = q * P.CDS_flaps / 2;
      loads.pointLoads.push({
        y: fr * P.span, Fz, Fx,
        torque: -(xa - sec.xSC) * Fz + (za - sec.zEA) * Fx,
      });
    }
  }
  const beam = beamStatics(P, sec, loads);

  const read = (g) => {
    const i = locate(sec, g.xc, g.surface);
    const s = sec.segs[i];
    // average along the grid: a 3 mm gauge samples a finite slice of the span
    const half = 0.5 * (g.gridLength ?? 3e-3) * Math.abs(Math.cos((g.angle || 0) * Math.PI / 180));
    const ns = half > 1e-6 ? 5 : 1;
    let Mx = 0, Mz = 0, T = 0, Sz = 0;
    for (let k = 0; k < ns; k++) {
      const yy = Math.max(0, Math.min(P.span,
        g.y + (ns === 1 ? 0 : -half + 2 * half * k / (ns - 1))));
      Mx += at(beam.y, beam.Mx, yy) / ns; Mz += at(beam.y, beam.Mz, yy) / ns;
      T += at(beam.y, beam.T, yy) / ns;   Sz += at(beam.y, beam.Sz, yy) / ns;
    }
    const eSpan = epsYY(sec, s.mid[0], s.mid[1], Mx, Mz);
    const eChord = -s.lam.nu_xy * eSpan;
    const qflow = T / (2 * sec.Aenc) + sec.qhat[i] * (-Sz);
    const gamma = qflow / (s.lam.Gxy * s.lam.thickness);
    const eGrid = resolveStrain(eSpan, eChord, gamma, g.angle || 0);
    const ratio = (BRIDGE[g.bridge] || 0.25) * (g.gf || 2.05) * eGrid;
    return { seg: s, i, eSpan, eChord, gamma, eGrid, ue: eGrid * 1e6,
             mVperV: ratio * 1e3, mV: ratio * (g.exc || 5) * 1e3, Mx, Mz, T };
  };

  const gauge = read(P.gauge);
  const strutEA = P.strutE * Math.PI *
    ((P.strutOD / 2) ** 2 - (P.strutOD / 2 - P.strutWall) ** 2);
  const strut = { ue: -beam.reaction / strutEA * 1e6, EA: strutEA,
                  mVperV: 0.65 * 2.05 * (-beam.reaction / strutEA) * 1e3 };

  return { af, skin, sec, beam, loads, gauge, strut, q, Dtot, Del, Dcar,
           area, dragTotal: loads.dragTotal, read, shape };
}
