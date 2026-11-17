# Published agents

Local trains write **gitignored** zips under `ml/runs/` (with `receipt.json` inside after `python -m fightml.train`). Only **named generations** live here.

| id | sim | status | one line |
|---|---|---|---|
| [`ppo_mix4`](ppo_mix4.yaml) | 0.4.1 | **current** | dummy-only 30e6; timeout win +0.25; KOs zoner/defensive |
| [`ppo_mix3`](ppo_mix3.yaml) | unknown | deprecated | jump-stall mix (timeout-at-0; mix4 fixed this) |
| [`ppo_mix2`](ppo_mix2.yaml) | unknown | deprecated | mash-heavy mix |
| [`ppo_mix`](ppo_mix.yaml) | unknown | deprecated | early personality mix |
| [`ppo_block`](ppo_block.yaml) | unknown | deprecated | first dummy (block) |

Schema for a card (`<id>.yaml` next to `<id>.zip`):

```yaml
id: ppo_mix5
status: published          # published | deprecated | broken
created: YYYY-MM-DD
author: <github handle>
game:
  simulator_version: "0.4.1"
  commit: "<earliest_game_commit from receipt.json>"
  tag: game-v0.4.1         # if tagged
ml:
  commit: "<earliest_ml_commit from receipt.json>"
python: "3.10.11"
lockfile: requirements-lock.txt
compat:
  sim: ["0.4.1"]
  obs: Phase1-flatten
  policy: MaskablePPO-MlpPolicy
train:
  command: "python -m fightml.train ..."
  timesteps: 30000000
  notes: "..."
eval_hint: "python -m fightml.eval artifacts/ppo_mix5 --p1 human"
```

When publishing:

1. Copy `ml/runs/<stem>.zip` here (keep the receipt).
2. `unzip -p ml/artifacts/<id>.zip receipt.json` and paste `git.head` / `earliest_game_commit` / `earliest_ml_commit` into the yaml.
3. Fill compat, opponents, notes. Point README “current policy” at the new id.
4. PR `ml/publish-<id>` — zip + yaml only, no `ml/runs/` pools.

Older zips in this folder have **no** `receipt.json` (`commit: unknown`). Do not treat them as evidence on a newer `SIMULATOR_VERSION` unless `compat.sim` says so.

```bash
python -m fightml.eval artifacts/ppo_mix4 --p1 human
unzip -p ml/artifacts/ppo_mix4.zip receipt.json   # empty until a post-receipt zip is published
```
