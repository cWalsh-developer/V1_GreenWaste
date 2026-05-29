# LCA Assumptions for V1

This project currently estimates indicative scenario-based CO2e ranges. It is not
a full product LCA and should not be presented as an exact product carbon
footprint.

## Scope

The V1 calculation starts after object recognition and approximate size
estimation:

1. The detector provides an item class and object region.
2. Depth data provides approximate size cues.
3. Reference matching estimates likely weight and material family.
4. Scenario factors convert the estimated weight range into indicative CO2e
   ranges.

The current calculation is closest to an end-of-life decision-support model. It
does not include a complete cradle-to-grave product inventory.

## Factor Source

The default factor table is:

`data/interim/mappings/lca_scenario_factors.csv`

Most factors are derived from the UK Government GHG Conversion Factors for
Company Reporting 2025. The original flat-file factors are published in
kg CO2e per tonne, so the project table converts them to kg CO2e per kg.

The default class-level material composition table is:

`data/interim/mappings/material_composition_profiles.csv`

This table provides low-impact and high-impact proxy material mixes for each
broad item class. These are not measured compositions. They are scenario profiles
used to carry material uncertainty through the calculation.

The reference-matching stage may also select a more specific profile when a
simple keyword rule supports it. For example, a broad `chair_seating` detection
can use the `office_chair` profile if the closest reference products contain
terms such as desk chair, office chair, gaming chair, computer chair, or swivel
chair.

## Scenario Meanings

- `reuse_avoided_production`: proxy for avoided production if the item is reused
  rather than replaced. This uses material-use factors and should be treated as a
  proxy, not a complete consequential LCA.
- `closed_loop_recycling`: end-of-life waste-disposal factor for closed-loop
  recycling where available.
- `incineration_energy_recovery`: end-of-life waste-disposal factor for
  incineration with energy recovery where available.
- `landfill`: end-of-life waste-disposal factor for landfill where available.

## Recommendation Logic

The LCA output includes all scenario results plus a recommended route. The
recommendation is selected by the lowest upper-bound CO2e estimate among the
modelled routes. This deliberately favours the option with the best conservative
CO2e outcome, but it does not account for operational constraints such as local
facility availability, contamination, safety, transport distance, or item
condition. Reuse is treated as a diversion route rather than literal disposal.
The user-facing recommended route is restricted to one of four categories:
Reuse, Landfill, Incineration, or Recycle.

## Limitations

- Material composition is inferred from reference products, not measured.
- Broad item classes use low/high proxy composition profiles.
- Some material-specific factors are unavailable, so proxies are used for foam,
  unknown material, and missing pathways.
- The result should be reported as an indicative CO2e range or comparative score.
- A full ISO-style LCA would require explicit goal/scope definition, life-cycle
  inventory, impact assessment method, interpretation, sensitivity analysis, and
  external review if making public comparative claims.
