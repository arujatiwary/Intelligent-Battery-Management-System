**Overview**

Conventional CC-CV chargers apply the same fixed current profile regardless of battery condition — a degraded cell at 40°C gets identical treatment as a healthy cell at 25°C, with no mechanism to prevent thermal damage.
This project presents a five-agent pipeline where every charging decision is uncertainty-aware and safety-verified:

A Probabilistic Transformer estimates SoC, SoH, and temperature with explicit confidence

A Simulator-Optimiser generates 60 Pareto-optimal charging policies via GA + NSGA-II

A Meta-Agent selects a policy based on battery health and predictor confidence

A Kill Agent evaluates the full simulated trajectory and issues allow / override / abort

A Final Command agent outputs the safe charging decision

Experiments show 6–9% lower SoH loss versus CC-CV 1C, and correct abort in thermally unsafe conditions where all CC-CV variants proceed uninhibited.



**Dataset**

NASA Li-ion Battery Dataset — Saha & Goebel, NASA Ames Prognostics Data Repository, 2007.
