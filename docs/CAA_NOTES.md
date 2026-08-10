# Notes on UK CAA rules for a real-flight extension

Everything in this repository is simulation. This page records what would have
to change to fly any of it outdoors in the UK, because "it works in SITL" and
"it is legal and safe to fly" are very different claims and the gap between them
is most of the work.

**This is engineering notes, not legal advice.** The rules change; check
[CAP 722](https://www.caa.co.uk/publication/download/16018) and the current
[CAA drone and model aircraft code](https://register-drones.caa.co.uk/drone-code)
before flying anything.

## Which category this would fall into

An inspection multirotor of the class modelled here — around 1.6 kg with a
camera, flown near a structure — does not fit the Open category comfortably:

- **A1** (over people) needs a sub-250 g aircraft or a C0/C1 class mark. A 1.6 kg
  inspection frame is neither.
- **A2** (close to people) needs a C2 class mark and an A2 Certificate of
  Competency, and still requires 30 m separation from uninvolved people (5 m in
  low-speed mode).
- **A3** (far from people) requires 150 m from residential, commercial,
  industrial or recreational areas. Most structures worth inspecting are inside
  one of those.

So realistic facade or mast inspection work is normally **Specific category**,
under an Operational Authorisation, which means a SORA-style risk assessment,
an operations manual, and demonstrated competence. That is a paperwork and
process exercise at least as large as the software one.

## What the software would have to gain

The gaps between this repository and something that could support a Specific
category application:

**Containment.** The geofence here is a single polygon plus an altitude band,
enforced by the same software that flies the aircraft. An operational
authorisation generally wants containment that is *independent* of the flight
control it protects — a second processor, an independent power source, or at
minimum a geofence enforced by the autopilot rather than by the companion
computer. See [`LIMITATIONS.md`](LIMITATIONS.md).

**Ground risk buffer.** The RTL implemented here climbs, transits straight home
and descends. It has no map, and it will fly over whatever is between it and
home. A real operation has to show where the aircraft ends up in every failure
case, including a total loss of thrust, and that computation needs the wind and
the height at the moment of failure.

**Loss of control link.** The failsafe matrix holds for 3 s and returns home
after 15 s. Those numbers are defensible but arbitrary; an operations manual has
to justify them against the actual link budget and the site.

**Electronic conspicuity and remote ID.** Not modelled at all. UK requirements
here are still settling, and anything flown commercially now needs to track
them.

**Logging for occurrence reporting.** The ULog output is the right format and
already contains what an investigation would want — commanded against achieved,
estimator health, failsafe triggers with cause and timestamp. What is missing is
tamper-evidence and a retention policy.

## Things this repository already does that would help

Not everything here is a gap. These would carry over:

- **The failsafe matrix is a table with stated rationale per row**, generated
  into [`FAILSAFE_MATRIX.md`](FAILSAFE_MATRIX.md) and tested row by row
  end-to-end. That is close to the form a hazard analysis wants, and the
  traceability from rule to test to document is the part that is usually
  missing.
- **Mission files are validated before flight**, including cross-field
  geometry: a mission whose waypoints lie outside its own geofence, or above its
  own ceiling, is rejected on the ground.
- **The geofence is predictive**, so it acts on where the aircraft will be
  rather than where it has already gone. Containment arguments care about
  overshoot, and a positional check has none of that margin.
- **Every quantitative claim is reproducible from a script**, which is the
  habit an assurance argument needs even though the specific numbers here are
  not the ones that would be submitted.

## RTK specifically

The estimator assumes an RTK correction stream. Outdoors that means either a
local base station or a network service (Ordnance Survey Net, or a commercial
NTRIP provider). Two consequences worth stating:

- **RTK is not a safety function here and should not become one.** The failsafe
  matrix degrades to a float solution without acting, and only responds when the
  fix is lost entirely. That is deliberate: a correction stream that depends on
  a mobile network is not something to hang containment on.
- **Base station siting matters more than the receiver.** Near a large steel
  structure, multipath dominates, and a fixed solution can be confidently wrong.
  The innovation gate and covariance check in the estimator exist partly for
  this, but they are a mitigation, not a solution.
