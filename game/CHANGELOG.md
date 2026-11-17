# fightsim changelog

Keep [`SIMULATOR_VERSION`](fightsim/config.py) in sync with [`pyproject.toml`](pyproject.toml). See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for when to bump.

**Old agents:** if a row says invalid, do not eval older `ml/artifacts/` zips on that version and treat the numbers as evidence. Retrain.

## 0.4.1

- **Date:** 2026-10-26 (tree at `c79dced`, “version 1.0 ready for handoff”).
- **What:** Current published meta: personalities, pulse (214), mix4-era combat (fireball chip/anti-camp, crouch-safe stand light, no blockstun pose swap). Render-only window/HUD does not affect `step()`.
- **Old agents:** mix4 was trained on this sim. Older zips (`ppo_block` … `ppo_mix3`) were trained on earlier metas; they may still *load* but are not evidence on 0.4.1.
