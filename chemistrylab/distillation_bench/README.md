# Distillation Bench
Modules defining the distillation bench

## Environments

- distillation_bench_v1 : Provide a vessel, assign a material to isolate, and initiate the distillation engine.
- distillation_bench_v1_engine : Use the distillation module, `../distillations/distill_v0`, to perform distillation actions on the vessel provided by the distillation environment.

## Overview

The distillation bench environment is intended to utilize the different boiling points between materials in a vessel and boil off materials sequentially into a secondary vessel. The boiled off materials are moved to a tertiary vessel until the secondary vessel contains only the material designated as the desired material.