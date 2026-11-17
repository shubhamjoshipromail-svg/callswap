"""Write report.json and a self-contained report.html, and format the terminal summary."""
import html
import json
from pathlib import Path

from .economics import economics

LIMITS = ('Agreement with the teacher labels is not correctness: if the teacher was wrong, the local model learned to be wrong the same way.',
          'Labels with few sealed examples are unproven, even when the overall verdict is PASS.',
          'Data drift is not measured: the result holds for inputs like the ones in this file.')

STYLE = ''':root{color-scheme:light dark}body{font:16px system-ui,sans-serif;max-width:900px;margin:auto;padding:1rem;line-height:1.5}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #8886;padding:.4rem;text-align:left}
.scroll{overflow:auto}svg{width:100%;height:auto}polyline{fill:none;stroke:#4688dc;stroke-width:3}
.verdict{display:inline-block;padding:.1rem .7rem;border-radius:.4rem;color:#fff}.PASS{background:#1a7f37}.FAIL{background:#b42318}
.cards{display:flex;flex-wrap:wrap;gap:.75rem}.card{flex:1 1 11rem;border:1px solid #8886;border-radius:.5rem;padding:.6rem .8rem}
.card b{display:block;font-size:1.5rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}'''


def pct(x, digits=1):
    return '—' if x is None else f'{x:.{digits}%}'


def summary(data):
    """Plain-text verdict for the terminal."""
    ev, econ = data['evaluation'], data['economics']
    a = econ['assumptions']
    lines = [
        f"callswap verdict: {ev['verdict']}   (bar: agreement lower bound >= {pct(ev['target_agreement'], 0)}, coverage >= {pct(ev['min_coverage'], 0)})",
        *[f'  why: {reason}' for reason in ev['fail_reasons']],
        f"  coverage              {pct(ev['coverage'])}   the local model answered {ev['accepted']:,} of {ev['sealed_rows']:,} sealed calls; the rest go to the LLM",
        f"  agreement             {pct(ev['agreement'], 2)}  on the calls it answered (95% lower bound {pct(ev['agreement_lower_bound'], 2)})",
        f"  without abstaining    {pct(ev['overall_accuracy'])}   agreement if the local model answered everything",
        f"  calls saved           {econ['calls_saved_per_1000']:.0f} of every 1,000",
        f"  estimated saving      ${econ['monthly_saving']:,.0f}/month at {a['monthly_calls']:,} calls/month, ${a['price_per_1k_calls']:.2f} per 1,000 calls, ${a['serving_cost']:.0f}/month serving"
        + ('  (no positive saving)' if econ['monthly_saving'] <= 0 else ''),
        f"  local latency         {ev['latency_ms']['p50']:.1f} ms median per call on this machine ({data['training']['winner']} model)",
    ]
    weak = [l for l in ev['weakest_labels'][:3] if ev['per_label'][l]['coverage'] < 1 or (ev['per_label'][l]['agreement'] or 1) < 1]
    if weak:
        lines.append('  weakest labels        ' + '; '.join(
            f"{l} (coverage {pct(ev['per_label'][l]['coverage'], 0)}, agreement {pct(ev['per_label'][l]['agreement'], 0)})" for l in weak))
    if ev['labels_unproven']:
        lines.append(f"  unproven labels       {len(ev['labels_unproven'])} label(s) have fewer than 20 sealed examples")
    if ev['verdict'] == 'FAIL' and ev['agreement_lower_bound'] >= ev['target_agreement']:
        lines.append(f"  reading this          the agreement bar is met on the calls the local model takes; it just takes fewer than "
                     f"--min-coverage. Sending those {pct(ev['coverage'], 0)} locally is still proven to the stated bound.")
    lines.append('  Agreement is with the labels in your file, not with ground truth.')
    return '\n'.join(lines)


def report(out, label_report, price=.5, monthly_calls=3000000, serving_cost=20):
    out = Path(out)
    load = lambda name: json.loads((out / name).read_text())  # noqa: E731
    manifest, ev, cal, train = load('manifest.json'), load('sealed_result.json'), load('calibration_result.json'), load('train_result.json')
    econ = economics(ev['coverage'], price, monthly_calls, serving_cost)
    data = {'evaluation': ev, 'economics': econ, 'manifest': manifest, 'calibration': cal, 'training': train,
            'label_report': label_report}
    (out / 'report.json').write_text(json.dumps(data, indent=2))

    esc = lambda s: html.escape(str(s))  # noqa: E731
    points = ' '.join(f"{40 + 560 * p['coverage']:.1f},{210 - 180 * p['agreement']:.1f}"
                      for p in sorted(cal['curve'], key=lambda p: p['coverage']))
    label_rows = ''.join(
        f"<tr><td>{esc(k)}</td><td>{v['count']}</td><td>{pct(v['coverage'])}</td><td>{pct(v['agreement'])}</td>"
        f"<td>{pct(v['agreement_lower_bound'])}</td></tr>" for k, v in ev['per_label'].items())
    mistakes = ''.join(
        f"<li>{esc(x['text'])} — teacher: {esc(x['teacher_label'])}; local: {esc(x['prediction'])}; confidence {x['confidence']:.4f}</li>"
        for x in ev['disagreements']) or '<li>None</li>'
    saving = f"${econ['monthly_saving']:,.0f}" + (' (no positive saving)' if econ['monthly_saving'] <= 0 else '')
    reasons = ''.join(f'<li>{esc(r)}</li>' for r in ev['fail_reasons'])
    training = {k: v for k, v in train.items() if k != 'labels'}
    listing = lambda items: esc(', '.join(items) or 'none')  # noqa: E731
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>callswap report: {ev['verdict']}</title><style>{STYLE}</style></head><body>
<h1>Verdict: <span class="verdict {ev['verdict']}">{ev['verdict']}</span></h1>
<p>Bar: agreement lower bound at least {pct(ev['target_agreement'], 0)} and coverage at least {pct(ev['min_coverage'], 0)}.</p><ul>{reasons}</ul>
<div class="cards"><div class="card">Coverage<b>{pct(ev['coverage'])}</b>{ev['accepted']:,} of {ev['sealed_rows']:,} sealed calls answered locally</div>
<div class="card">Agreement, 95% lower bound<b>{pct(ev['agreement_lower_bound'], 2)}</b>observed {pct(ev['agreement'], 2)} on answered calls</div>
<div class="card">Estimated monthly saving<b>{saving}</b>{econ['calls_saved_per_1000']:.0f} of every 1,000 calls stay local</div></div>
<h2>What was tested</h2>
<p>Agreement between the local model and the labels in the input file, on a sealed split that training, model selection,
calibration and threshold choice never read. Split ({esc(manifest.get('method', 'stratified'))}): {esc(manifest['counts'])}.
Sealed ids and labels SHA-256: <code>{manifest['sealed_ids_labels_sha256']}</code>. The sealed set was used once.</p>
<p>Agreement if the local model answered every call (no abstaining): {pct(ev['overall_accuracy'])}.
Local latency: {ev['latency_ms']['p50']:.1f} ms median, {ev['latency_ms']['p95']:.1f} ms 95th percentile per call on the machine that ran this report.</p>
<h2>Training and model selection</h2><pre>{esc(json.dumps(training, indent=2))}</pre>
<h2>Coverage against agreement (calibration split)</h2>
<svg viewBox="0 0 640 240" role="img" aria-label="Calibration coverage against agreement"><path d="M40 30 V210 H600" stroke="currentColor" fill="none"/><polyline points="{points}"/><text x="250" y="238" fill="currentColor">Coverage (0–100%)</text><text x="45" y="20" fill="currentColor">Agreement (0–100%)</text></svg>
<p>Confidence threshold {cal['threshold']:.4f}, temperature {cal['temperature']:.3f}. Target met on the calibration split: {cal['target_met_on_calibration']}.</p>
<h2>Per-label sealed results</h2><div class="scroll"><table><tr><th>Label</th><th>Sealed rows</th><th>Coverage</th><th>Agreement</th><th>95% lower bound</th></tr>{label_rows}</table></div>
<h2>Teacher inconsistencies and rare labels</h2>
<p>Rows whose text appears with more than one label: {label_report['teacher_inconsistency_rows']}.
Too rare to prove (fewer than 20 rows in the input): {listing(label_report['rare_labels'])}.
Fewer than 20 sealed rows: {listing(ev['labels_unproven'])}.
Absent from the sealed split: {listing(ev['labels_missing_from_sealed'])}.</p>
<h2>Lowest-confidence accepted disagreements</h2><ul>{mistakes}</ul>
<h2>Limits</h2><ul>{''.join(f'<li>{esc(x)}</li>' for x in LIMITS)}</ul>
<h2>Savings arithmetic</h2><p>Baseline ${econ['baseline']:,.2f}/month; after ${econ['after']:,.2f}/month; saving {econ['percent_saving']:.1f}%;
break-even volume {esc(f"{econ['break_even_monthly_calls']:,.0f} calls/month" if econ['break_even_monthly_calls'] else 'not reachable')}.
Assumes ${price:.2f} per 1,000 LLM calls, {monthly_calls:,} calls a month and ${serving_cost:.0f} a month to serve the local model.</p>
</body></html>'''
    (out / 'report.html').write_text(content, encoding='utf-8')
    return data
