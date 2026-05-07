# Paper 2 Claims

## Primary Claim

Internal class-conditioning objectives are insufficient evidence of class-faithful synthetic malware image generation.

## Evidence

Across baseline cDCGAN, ACGAN auxiliary-loss, and fake-class-head ACGAN interventions, the external real-only classifier audit detected severe class-conditioning failure.

## Seed-Stability Evidence

For the fake-class-head ACGAN intervention, seeds 42, 43, and 44 all failed external conditioning audits:

- Seed 42: full collapse to class 7
- Seed 43: collapse into classes 2, 5, and 8
- Seed 44: full collapse to class 8

The exact collapse mode varied by seed, but the conditioning failure persisted.

## Secondary Claim

Downstream utility and class-conditional semantic fidelity are distinct evaluation targets.

The fake-class-head ACGAN runs showed positive downstream utility deltas in some cases, even while external conditioning audits showed severe class collapse. This means synthetic samples may improve classifier performance without being class-faithful to their requested labels.

## Paper Positioning

Paper 2 should argue for external conditioning audits as a necessary part of evaluating class-conditional synthetic cybersecurity image generators.