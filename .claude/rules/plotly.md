---
description: 'Plotly conventions: plotly.express first with graph_objects for fine control, batch update_layout and update_traces styling, and static export via kaleido. Apply when writing or reviewing interactive figures, dashboards, or plotly figure-generation code.'
paths:
- '**/*.py'
provenance: shared/rules/stack/plotly.md @ 85dde8d
---

You are an expert in plotly figure construction.

## Principles

1. Start with `plotly.express`; drop to `graph_objects` only when express cannot say it.
2. Style through batch updates on the figure, never by rebuilding trace dicts by hand.

## Directives

- Build standard charts with `px.*` from long-form frames, mapping columns to `x`/`y`/`color`/`facet_*`; reshape data before plotting instead of looping `add_trace`.
- Refine with `fig.update_layout`, `fig.update_traces`, and `fig.update_*axes` (selectors for per-trace targeting), keeping construction and styling as separable steps.
- Mix express and graph_objects by adding `go` traces to an express figure, never by reconstructing the express output manually.
- Export static images with `fig.write_image` (requires `kaleido`) and interactive output with `fig.write_html(include_plotlyjs="cdn")`; choose per artifact, with static formats for committed or published figures.
- In notebooks and apps, return or display the figure object; reserve `fig.show()` for ad-hoc local inspection.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `import plotly.plotly` / `py.iplot` (Chart Studio era) | `fig.show()`, `fig.write_html`, `fig.write_image` |
| `plotly.offline.plot` in new code | `fig.write_html` |
| `fig["layout"]["xaxis"]["title"] = ...` dict surgery | `fig.update_xaxes(title_text=...)` |
| `orca` for static export | `kaleido` |
