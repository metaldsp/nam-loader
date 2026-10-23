<!--
SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini

SPDX-License-Identifier: MIT
-->

# Calibration and loudness

A `Runtime` puts gain around its model in two places: before the model it can
drive the capture at the level of its original rig (input calibration), and
after the model it applies either a calibrated hand-off or loudness
normalization. This page explains what each one does, why the last two are
mutually exclusive, and how a host maps its own routing onto the
`ProcessContext` fields that control them.

For the vocabulary, see [CONTEXT.md](../CONTEXT.md).

## The two kinds of metadata

A `.nam` file may carry reference levels and a loudness figure. `loadModel`
exposes them on `Model` as optional values, absent when the file does not
carry them:

| Field | Meaning |
| --- | --- |
| `inputLevelDbu` | The level (dBu) that drives the captured gear to the capture's reference input. |
| `outputLevelDbu` | The level (dBu) the captured gear produced at that reference input. |
| `loudnessDb` | The capture's measured loudness, for normalization toward a common target. |

Together `inputLevelDbu` and `outputLevelDbu` are the model calibration: they
describe the physical levels of the captured gear. Each is optional on its own,
and the `Runtime` checks them independently
(`Model.EachLevelIsReportedIndependently`, `Model.CaptureWithoutLevelsHasNone`).

The host's own setting is the input calibration: the interface-level reference
that relates digital full scale to the analog level the models expect, in dBu
(`ProcessContext::inputCalibrationDbu`, default 12 dBu). It describes the
user's interface, not any capture, so it belongs with the host's session
settings rather than with a model.

## Input calibration

With `ProcessContext::calibrateInput` set and a model that has
`inputLevelDbu`, the `Runtime` applies a gain before the model:

```
gain (dB) = inputCalibrationDbu - model.inputLevelDbu
```

This drives the capture at the level it was made at, so the model sees the
signal level its rig saw. A model without an input level is left at
unity (`Runtime.InputCalibrationDrivesTheModelAtItsCaptureLevel`). Unlike
normalization, this gain is not smoothed: it changes the signal the model
sees, so it follows the context on every call.

Input calibration is a property of the signal path into the model, so it also
applies when a model follows another model. That is what makes the hand-off
below work.

## What happens after the model

After the model the `Runtime` applies one gain, chosen from two mutually
exclusive options. The options pursue contradictory goals:

- The **calibrated hand-off** reproduces the capture rig's true output level,
  so a later model is driven the way a real device chained to the first would
  drive it.
- **Loudness normalization** makes every capture equally loud, whatever level
  it was captured at.

Applying both would undo the first with the second, so the `Runtime` never
does. The hand-off wins whenever its conditions hold; normalization is the
fallback.

### Calibrated hand-off

The hand-off applies when all of these hold:

1. `calibrateInput` is set.
2. `calibratedHandOff` is set. This is how the host says that another enabled
   model follows this one on the signal path.
3. The model has an `outputLevelDbu` that is finite and within +/-60 dBu.

The gain is then:

```
gain (dB) = model.outputLevelDbu - inputCalibrationDbu
```

That converts the output back into the host's analog reference frame. The
next model's own input calibration (`inputCalibrationDbu - nextInputLevelDbu`)
converts from that frame into its own, so the host's calibration setting
cancels. What is left is `outputLevelDbu - nextInputLevelDbu`: exactly the
level difference of plugging device A into device B
(`Runtime.HandoffIsOutputLevelMinusCalibration`).

Two properties follow:

- The hand-off is not clamped. It is a physical level difference, not an
  estimate, so a capture that really did come out 20 dB hotter hands on 20 dB.
- It applies even if `normalize` is false: the hand-off is not normalization.

The +/-60 dBu check is a sanity check on the metadata. A value outside the
range (or NaN/infinite) is treated as broken, and the model falls back to
normalization instead of applying a wild gain
(`Runtime.OutOfRangeOutputLevelFallsBackToNormalization`). The same fallback
applies when the model has no `outputLevelDbu` or either flag is unset
(`Runtime.HandoffNeedsTheFlagAndAnOutputLevel`,
`Runtime.HandoffFallsBackToNormalizationWhenItsConditionsFail`).

The last model on every path never hands off. Calibrated output at the end of a
path would make the overall volume swing with each capture's metadata, which
is exactly what normalization is for. The host guarantees this by leaving
`calibratedHandOff` false for terminal models.

### Loudness normalization

When the hand-off does not apply and `ProcessContext::normalize` is set, the
gain moves the model's loudness to a common target:

```
gain (dB) = clamp(targetLoudnessDb - model.loudnessDb, -12, +12)
```

- The target is `ProcessContext::targetLoudnessDb` (default -18 dB).
- The adjustment is clamped to +/-12 dB. Loudness metadata is a measurement
  of one capture session, not a guarantee; the clamp bounds the damage of a
  wrong figure (`Runtime.NormalizationIsClampedToPlusMinusTwelveDb`).
- A model with no `loudnessDb`, or one that is NaN, infinite or outside
  -100..0 dB, is treated as already at the target, so its gain is 0 dB
  (`Runtime.OutOfRangeLoudnessFallsBackToTheTarget`).
- With `normalize` false the gain is unity
  (`Runtime.NormalizeOffRemovesTheNormalizationGain`).

`Runtime.NormalizationMovesLoudnessToTheTarget` shows the basic behaviour.

If neither the hand-off nor normalization applies, the post-model gain is
unity.

## Smoothing and first-block priming

The post-model gain is smoothed with a linear ramp over 50 ms, counted at the
oversampled rate. Changing a flag between calls (powering a following model
on or off, for example) therefore glides rather than steps
(`Runtime.GainChangesGlideOverFiftyMilliseconds`). Because the choice is a
per-call flag, toggling it never rebuilds anything
(`Runtime.HandOffFlagToggleSwitchesWithoutRebuild`).

The first block a new `Runtime` processes is special: the smoother snaps to
the gain the block asks for instead of ramping from unity. A replacement
runtime that a host crossfades in next to a running one would otherwise glide
in from 0 dB while the old one already holds its final gain, and the blend
would be audibly wrong during the fade
(`Runtime.FirstBlockStartsAtTheGainItWillHold`). See
[hosting.md](hosting.md) for the crossfade.

The input-calibration gain is not part of this smoothing.

## Mapping onto `ProcessContext`

| Field | Set by the host to... |
| --- | --- |
| `calibrateInput` | The user's "input calibration" switch. |
| `inputCalibrationDbu` | The user's interface reference level, in dBu. |
| `normalize` | The user's loudness-normalization switch (per model if you offer that). |
| `targetLoudnessDb` | The common listening level to normalize toward. |
| `calibratedHandOff` | True when another enabled model follows this one on the signal path and `calibrateInput` is on. |

The first four are settings. `calibratedHandOff` is topology: the library does
not know the host's routing, so the host decides it per model and passes it on
every `process` call. The rule is:

- **True** when `calibrateInput` is on and at least one other model that is
  currently enabled (powered, loaded) is reachable downstream of this one.
- **False** otherwise, which includes every model that is the last one on all
  of its paths. These normalize.

Disabling the downstream model flips the flag, and the gain glides over
50 ms; nothing is rebuilt.

The `Runtime` itself checks the metadata (the +/-60 dBu rule); the host only
answers "is there a following model?".

## Worked example: Ampira

Ampira, the library's original host, is a Chain of Blocks with parallel
sections: a Split sends the signal down Branch A and Branch B, and a Merge sums
them again. It keeps a table of downstream models for each model and builds
`calibratedHandOff` from it. Its rules:

- **Reachability.** A model on Branch A reaches the later blocks of Branch A
  and everything after the Merge, never Branch B. A model after a Merge is
  downstream of the models of both branches. The table ignores power and load
  state; the host checks those on each call, so powering a downstream block
  on or off changes the gain without rebuilding anything.
- **A shared model feeding two paths keeps one level.** A model before a Split
  hands off once, with one calibrated output level for both branches. There are
  no per-branch gains inside the Split; the branch without a model can be
  trimmed with its own controls.
- **Terminal models normalize individually.** A model with nothing usable
  downstream normalizes, clamped to +/-12 dB. Two terminal models on separate
  branches each normalize before the Merge.
- **Missing metadata falls back to normalization.** This is the `Runtime`'s own
  fallback, not extra host logic.
- **Intentional gain changes are not compensated.** Split A/B positioning,
  crossover filtering, block gains and Merge levels are the user's choices and
  keep driving whatever model follows.

Another host, with a different routing model, only needs to answer the same
question for each of its models: is there another enabled model after me?
