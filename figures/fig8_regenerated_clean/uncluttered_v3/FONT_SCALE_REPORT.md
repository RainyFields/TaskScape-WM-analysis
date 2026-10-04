# Manuscript-width type check

## Width used

The manuscript `.tex` is not in this repository. Nature Scientific Data places
full-width 3-panel figures in `figure*` with

```latex
\includegraphics[width=\textwidth]{...}
```

The PDF canvas is **364 × 296 mm** (wider than
Nature `\textwidth` = 183 mm) with 16 mm between
stacked panels.

| Quantity | Value |
|---|---|
| PDF MediaBox | 366.96 × 296.00 mm |
| `figsize` | 14.3307 × 11.6535 in |
| Scale at native canvas size | 1.000 |
| Scale if included at `width=\textwidth` (183 mm) | 0.499 |

## Effective type after scaling (pt)

| Element | Drawn | Effective |
|---|---:|---:|
| Panel titles | 15.5 | 15.50 |
| Shared y-axis label | 14.5 | 14.50 |
| y / x tick labels | 12.0 | 12.00 |
| InterDMS sequence row | 11.5 | 11.50 |
| Family group labels | 12.0 | 12.00 |
| Legend titles | 13.0 | 13.00 |
| Legend entries | 11.5 | 11.50 |
| Panel letters | 16.5 | 16.50 |

**Smallest effective font size after scaling: 11.50 pt** (legend entries / sequence row).

## Caption / Methods / Results wording

- Caption, Methods, and Results all say **50 random split-half iterations**, not bootstraps.
- Within-subject: diagonal of the 5 × 5 matrix (each participant’s half A vs half B).
- Across-subject pair: \(0.5[r(i_A,j_B)+r(j_A,i_B)]\); plotted subject-level value is the mean of the four pairs that include that participant.
- Boxes summarise the five subject-level estimates. Negative *r* retained. No *p*-values.

48 negative subject-level points. Panel ylims: {'vertex': [-0.8863860579147935, 1.12], 'region': [-0.4712714666813612, 1.12], 'network': [-0.5447148626446724, 1.12]}.
