import os
import sys
import json
import numpy as np
import faiss
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# FIX for Windows OS Error 1455 (Paging file too small) during safetensors mmap
os.environ["SAFETENSORS_FAST_DISABLE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# LangChain Imports
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.output_parsers import StrOutputParser
from typing import Any, List, Optional

import config  # Our newly refactored config file

try:
    import formulas as ScientificEngine
except ImportError:
    print("[ERROR] formulas.py not found in the current directory.")
    sys.exit(1)


# ==========================================
# 1. CUSTOM LANGCHAIN LLM FOR LORA SWITCHING
# ==========================================
class LoRASwitchingLLM(LLM):
    """Custom LangChain LLM that handles dynamic PEFT adapter switching on the fly."""
    model: Any
    tokenizer: Any
    adapter_name: str = "hypothesis"

    def set_adapter(self, adapter_name: str):
        self.adapter_name = adapter_name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        
        self.model.set_adapter(self.adapter_name)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=100)
            
        # SLICE out the prompt tokens so we ONLY decode the AI's new response
        generated_tokens = output[0][inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return decoded.strip()

    @property
    def _llm_type(self) -> str:
        return "lora_switching_local_llm"


# ==========================================
# 2. DISCOVERY REPORTER (Visuals & Markdown)
# ==========================================
# ── Chart style helpers ───────────────────────────────────────────────────────
def _get_colors(theme='dark'):
    if theme == 'light':
        return {
            'bg': '#ffffff', 'panel': '#f6f8fa', 'grid': '#d0d7de',
            'text': '#24292f', 'title': '#0969da', 'desc': '#57606a'
        }
    return {
        'bg': '#0b0e14', 'panel': '#161a22', 'grid': '#2a2f3a',
        'text': '#c9d1d9', 'title': '#58a6ff', 'desc': '#6e7681'
    }

def _make_fig(title: str, theme='dark'):
    """Create a styled single-chart figure."""
    c = _get_colors(theme)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(c['bg'])
    ax.set_facecolor(c['panel'])
    ax.grid(color=c['grid'], linestyle='--', linewidth=0.6, zorder=0)
    ax.tick_params(colors=c['text'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(c['grid'])
    ax.set_title(title, color=c['title'], fontsize=13, fontweight='bold', pad=14)
    plt.subplots_adjust(bottom=0.15, top=0.88)
    return fig, ax, c

def _save(fig, path: str):
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


class DiscoveryReporter:
    @staticmethod
    def generate_dashboard(props, comp, report_id, output_dir=None, targets=None):
        out_dir = output_dir if output_dir else config.GRAPHS_DIR
        os.makedirs(out_dir, exist_ok=True)
        paths = []

        desc_1 = (
            "Mole fraction of each alloying element in the proposed composition. "
            "The proportions directly govern phase stability, density, and mechanical response."
        )
        desc_2 = (
            "X-axis: Theoretical density (g/cm³) — denser alloys are heavier. "
            "Y-axis: Estimated UTS (MPa) — higher is stronger. "
            "The proposed alloy (red ★) is compared against common structural material families."
        )
        desc_3 = (
            "X-axis: Engineering strain (dimensionless, 0 → fracture). "
            "Y-axis: Engineering stress (MPa). "
            "Derived from the Ramberg-Osgood power-law model using the estimated UTS and elongation."
        )
        desc_4 = (
            "Horizontal bar chart showing key performance indicators normalised to typical maximums. "
            "Red dashed lines (◆) indicate the industry standard target for this application. "
            "PREN ≥ 25 = excellent corrosion resistance. Pugh Ratio > 1.75 = ductile behaviour."
        )

        for theme in ['dark', 'light']:
            # ── Chart 1: Elemental Composition (Donut) ────────────────────────
            fig, ax, c = _make_fig("Elemental Composition (%)", theme)
            ax.axis('off')

            # Split into major (≥ 1%) and trace (< 1%) elements
            major = {k: v for k, v in comp.items() if v >= 0.01}
            trace = {k: v for k, v in comp.items() if 0 < v < 0.01}
            
            plot_labels = list(major.keys())
            plot_values = list(major.values())
            
            if trace:
                trace_sum = sum(trace.values())
                trace_names = ", ".join(trace.keys())
                plot_labels.append(f"Trace ({trace_names})")
                plot_values.append(trace_sum)

            colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(plot_labels)))
            
            def autopct_fmt(pct):
                if pct < 0.1:
                    return '<0.1%'
                return f'{pct:.1f}%'

            # Build legend labels: "Fe — 76.4%" or "Trace (Mo, Si) — <0.1%"
            legend_labels = []
            for lbl, val in zip(plot_labels, plot_values):
                pct = val * 100 / sum(plot_values) if sum(plot_values) > 0 else 0
                if pct < 0.1:
                    legend_labels.append(f"{lbl}  —  <0.1%")
                else:
                    legend_labels.append(f"{lbl}  —  {pct:.1f}%")

            wedges, texts, autotexts = ax.pie(
                plot_values, labels=None,
                autopct=autopct_fmt, colors=colors,
                pctdistance=0.82, startangle=90,
                wedgeprops={'linewidth': 1.2, 'edgecolor': c['bg']}
            )
            ax.add_artist(plt.Circle((0, 0), 0.60, fc=c['bg']))
            for a in autotexts:  a.set_color('white'); a.set_fontsize(8)

            ax.legend(wedges, legend_labels, title="Elements",
                      loc='center left', bbox_to_anchor=(0.92, 0.5),
                      fontsize=8, title_fontsize=9,
                      frameon=True, facecolor=c['panel'], edgecolor=c['grid'],
                      labelcolor=c['text'])
            fig.subplots_adjust(left=0.0, right=0.72)

            p = os.path.join(out_dir, f"chart_composition_{report_id}_{theme}.png")
            _save(fig, p)
            if theme == 'dark': paths.append({"path": f"chart_composition_{report_id}.png", "caption": desc_1})

            # ── Chart 2: Ashby Plot (UTS vs Density) ──────────────────────────
            fig, ax, c = _make_fig("Ashby Plot — Ultimate Tensile Strength vs. Density", theme)
            benchmarks = {
                'Mild Steel':   (7.85, 450),
                'AISI 4340':    (7.85, 1470),
                'Al 7075-T6':   (2.81, 572),
                'Ti-6Al-4V':    (4.43, 950),
                'Inconel 718':  (8.19, 1375),
            }
            proposed_d = props.get('Theoretical_Density_g_cm3', 7.8)
            proposed_u = props.get('Estimated_UTS_MPa', 800)
            for name, (d, u) in benchmarks.items():
                ax.scatter(d, u, s=90, color='#555e6b', zorder=3)
                ax.annotate(name, (d, u), textcoords='offset points', xytext=(6, 4), fontsize=7.5, color=c['text'])
            ax.scatter(proposed_d, proposed_u, s=260, color='#e74c3c', marker='*', zorder=5)
            ax.annotate('Proposed', (proposed_d, proposed_u), textcoords='offset points', xytext=(8, 6), fontsize=8.5, color='#e74c3c', fontweight='bold')
            ax.set_xlabel('Density  (g/cm³)', color=c['text'], fontsize=10)
            ax.set_ylabel('Ultimate Tensile Strength  (MPa)', color=c['text'], fontsize=10)
            p = os.path.join(out_dir, f"chart_ashby_{report_id}_{theme}.png")
            _save(fig, p)
            if theme == 'dark': paths.append({"path": f"chart_ashby_{report_id}.png", "caption": desc_2})

            # ── Chart 3: Predicted Stress-Strain Curve ────────────────────────
            fig, ax, c = _make_fig("Predicted Stress–Strain Curve", theme)
            uts   = props.get('Estimated_UTS_MPa', 800)
            elong = props.get('Estimated_Elongation_pct', 15) / 100
            strain     = np.linspace(0, elong, 300)
            yield_str  = uts * 0.85
            stress     = np.where(strain < 0.002, yield_str * (strain/0.002),
                                  yield_str + (uts - yield_str) * (1 - np.exp(-40*(strain - 0.002))))
            if elong > 0:
                ax.scatter([elong], [stress[-1]], color='#e74c3c', s=80, zorder=5)
                ax.annotate('Fracture', (elong, stress[-1]), textcoords='offset points', xytext=(-20, 10),
                            fontsize=7.5, color='#e74c3c', arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=0.8))
            ax.plot(strain, stress, color='#2ecc71', linewidth=2.5, zorder=4)
            ax.fill_between(strain, stress, alpha=0.08, color='#2ecc71')
            ax.set_xlabel('Engineering Strain  (mm/mm)', color=c['text'], fontsize=10)
            ax.set_ylabel('Engineering Stress  (MPa)', color=c['text'], fontsize=10)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            p = os.path.join(out_dir, f"chart_stress_strain_{report_id}_{theme}.png")
            _save(fig, p)
            if theme == 'dark': paths.append({"path": f"chart_stress_strain_{report_id}.png", "caption": desc_3})

            # ── Chart 4: Normalized Property Performance (Horizontal Bar) ─────
            fig, ax, c = _make_fig("Mechanical & Corrosion Property Summary", theme)
            labels = ['PREN (Corrosion)', 'UTS (MPa)', 'Elongation (%)', 'Specific Strength', 'Pugh Ratio']
            
            abs_values = [
                props.get('PREN_Corrosion_Index', 0),
                props.get('Estimated_UTS_MPa', 0),
                props.get('Estimated_Elongation_pct', 0),
                props.get('Specific_Strength_MPa_cm3_g', props.get('Estimated_UTS_MPa', 0) / max(props.get('Theoretical_Density_g_cm3', 7.8), 0.1)),
                props.get('Pugh_Ratio', 1.75)
            ]
            
            norm_values = [
                min(abs_values[0] / 50.0, 1.0),
                min(abs_values[1] / 2000.0, 1.0),
                min(abs_values[2] / 50.0, 1.0),
                min(abs_values[3] / 250.0, 1.0),
                min(abs_values[4] / 3.0, 1.0)
            ]
            
            y_pos = np.arange(len(labels))
            bar_colors = ['#e74c3c', '#3498db', '#f1c40f', '#2ecc71', '#9b59b6']
            bars = ax.barh(y_pos, norm_values, align='center', color=bar_colors, height=0.6, zorder=3)
            
            for i, bar in enumerate(bars):
                val_str = f"{abs_values[i]:.1f}"
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                        val_str, va='center', color=c['text'], fontweight='bold', fontsize=9)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, color=c['text'], fontsize=9)
            ax.set_xlabel('Relative Performance to Typical Maximum', color=c['text'], fontsize=9)
            ax.set_xlim(0, 1.15)
            ax.set_xticklabels([])
            ax.invert_yaxis()  # top-to-bottom

            # ── Industry Target Markers ────────────────────────────────────
            if targets:
                norm_maxes = [50.0, 2000.0, 50.0, 250.0, 3.0]
                target_keys = ['PREN', 'UTS', 'Elongation', 'Specific_Strength', 'Pugh_Ratio']
                first_drawn = True
                for idx, key in enumerate(target_keys):
                    target_val = targets.get(key)
                    if target_val is not None:
                        norm_target = min(target_val / norm_maxes[idx], 1.0)
                        # Dashed vertical line spanning the bar height
                        ax.plot([norm_target, norm_target], [idx - 0.35, idx + 0.35],
                                color='#ff4757', linewidth=2, linestyle='--', zorder=6,
                                label='Industry Target' if first_drawn else None)
                        # Diamond marker at centre of bar
                        ax.scatter([norm_target], [idx], marker='D', color='#ff4757',
                                   s=30, zorder=7, edgecolors='white', linewidths=0.5)
                        # Small label above the bar
                        unit = {'PREN': '', 'UTS': ' MPa', 'Elongation': '%',
                                'Specific_Strength': '', 'Pugh_Ratio': ''}[key]
                        ax.text(norm_target, idx - 0.42, f'{target_val:.0f}{unit}',
                                color='#ff4757', fontsize=7, ha='center', fontweight='bold')
                        first_drawn = False
                if not first_drawn:
                    ax.legend(fontsize=8, loc='lower right',
                              facecolor=c['panel'], edgecolor=c['grid'], labelcolor=c['text'])
            
            p = os.path.join(out_dir, f"chart_properties_{report_id}_{theme}.png")
            _save(fig, p)
            if theme == 'dark': paths.append({"path": f"chart_properties_{report_id}.png", "caption": desc_4})

        return paths

    @staticmethod
    def generate_thermo_dashboard(thermo_data, report_id, output_dir=None):
        out_dir = output_dir if output_dir else config.GRAPHS_DIR
        os.makedirs(out_dir, exist_ok=True)
        paths = []

        desc_na = (
            "The PyCalphad simulation could not run — the TDB database may not contain "
            "all elements in this alloy, or the composition is outside the valid range."
        )
        desc_t1 = (
            "X-axis: Temperature (K). Y-axis: Mole fraction of each stable phase (0 = absent, 1 = fully dominant). "
            "Phase transitions appear as intersecting lines. BCC = ferrite; FCC = austenite; LIQUID = melt."
        )
        desc_t2 = (
            "X-axis: Temperature (K). Y-axis: Cₚ (J mol⁻¹ K⁻¹). "
            "Peaks indicate phase transitions (latent heat). Higher Cₚ means more energy needed to raise temperature — "
            "important for welding, casting, and heat-treatment design."
        )
        desc_t3 = (
            "X-axis: Temperature (K). Y-axis: Hₘ (J mol⁻¹). "
            "Steps/discontinuities mark first-order phase transitions (e.g. melting). "
            "The slope of each segment equals Cₚ for that phase."
        )
        desc_t4 = (
            "X-axis: Temperature (K). Y-axis: Chemical potential μ (J mol⁻¹). "
            "μ drives atomic diffusion: when two elements have equal μ across a phase boundary the system is at equilibrium. "
            "Diverging lines indicate increased thermodynamic driving force for segregation."
        )

        for theme in ['dark', 'light']:
            if not thermo_data:
                fig, ax, c = _make_fig("Thermodynamic Data Unavailable", theme)
                ax.text(0.5, 0.5, 'Calphad simulation skipped\n(incompatible TDB or out-of-range composition)',
                        ha='center', va='center', color='#e74c3c', fontsize=12, transform=ax.transAxes)
                p = os.path.join(out_dir, f"chart_thermo_unavailable_{report_id}_{theme}.png")
                _save(fig, p)
                if theme == 'dark': return [{"path": f"chart_thermo_unavailable_{report_id}.png", "caption": desc_na}]
                continue

            temps = thermo_data['temperatures']

            # ── Thermo 1: Phase Fractions ──────────────────────────────────────
            fig, ax, c = _make_fig("Phase Fraction vs. Temperature", theme)
            cmap = plt.cm.tab10(np.linspace(0, 1, len(thermo_data['phase_fractions'])))
            for (phase, fractions), col in zip(thermo_data['phase_fractions'].items(), cmap):
                ax.plot(temps, fractions, label=phase, linewidth=2, color=col)
            ax.set_xlabel('Temperature  (K)', color=c['text'], fontsize=10)
            ax.set_ylabel('Mole Fraction  (–)', color=c['text'], fontsize=10)
            ax.set_xlim(temps[0], temps[-1])
            ax.set_ylim(-0.02, 1.05)
            ax.legend(fontsize=8, facecolor=c['panel'], edgecolor=c['grid'], labelcolor=c['text'])
            p = os.path.join(out_dir, f"chart_phase_fractions_{report_id}_{theme}.png")
            _save(fig, p)
            if theme == 'dark': paths.append({"path": f"chart_phase_fractions_{report_id}.png", "caption": desc_t1})

            # ── Thermo 2: Molar Heat Capacity (Cp) ────────────────────────────
            fig, ax, c = _make_fig("Molar Heat Capacity  Cₚ  vs. Temperature", theme)
            ax.plot(temps, thermo_data['cpm'], color='#f1c40f', linewidth=2.2)
            ax.fill_between(temps, thermo_data['cpm'], alpha=0.10, color='#f1c40f')
            ax.set_xlabel('Temperature  (K)', color=c['text'], fontsize=10)
            ax.set_ylabel('Cₚ  (J mol⁻¹ K⁻¹)', color=c['text'], fontsize=10)
            ax.set_xlim(temps[0], temps[-1])
            p = os.path.join(out_dir, f"chart_heat_capacity_{report_id}_{theme}.png")
            _save(fig, p)
            if theme == 'dark': paths.append({"path": f"chart_heat_capacity_{report_id}.png", "caption": desc_t2})

            # ── Thermo 3: Molar Enthalpy (Hm) ─────────────────────────────────
            fig, ax, c = _make_fig("Molar Enthalpy  Hₘ  vs. Temperature", theme)
            ax.plot(temps, thermo_data['hm'], color='#e74c3c', linewidth=2.2)
            ax.fill_between(temps, thermo_data['hm'], alpha=0.10, color='#e74c3c')
            ax.set_xlabel('Temperature  (K)', color=c['text'], fontsize=10)
            ax.set_ylabel('Hₘ  (J mol⁻¹)', color=c['text'], fontsize=10)
            ax.set_xlim(temps[0], temps[-1])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.1f}k'))
            p = os.path.join(out_dir, f"chart_enthalpy_{report_id}_{theme}.png")
            _save(fig, p)
            if theme == 'dark': paths.append({"path": f"chart_enthalpy_{report_id}.png", "caption": desc_t3})

            # ── Thermo 4: Chemical Potentials (μ) ─────────────────────────────
            fig, ax, c = _make_fig("Chemical Potentials  μ  vs. Temperature", theme)
            cmap = plt.cm.Set2(np.linspace(0, 1, len(thermo_data['mu'])))
            for (element, mu_vals), col in zip(thermo_data['mu'].items(), cmap):
                ax.plot(temps, mu_vals, label=element, linewidth=2, color=col)
            ax.set_xlabel('Temperature  (K)', color=c['text'], fontsize=10)
            ax.set_ylabel('μ  (J mol⁻¹)', color=c['text'], fontsize=10)
            ax.set_xlim(temps[0], temps[-1])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.1f}k'))
            ax.legend(fontsize=8, facecolor=c['panel'], edgecolor=c['grid'], labelcolor=c['text'])
            p = os.path.join(out_dir, f"chart_chem_potentials_{report_id}_{theme}.png")
            _save(fig, p)
            if theme == 'dark': paths.append({"path": f"chart_chem_potentials_{report_id}.png", "caption": desc_t4})

        return paths

    @staticmethod
    def generate_markdown_report(goal, raw_ai, comp, props, analysis, report_id, chart_paths=None, thermo_paths=None, output_dir=None, thermo_insights=None, iteration_history=None):
        out_dir = output_dir if output_dir else os.path.join(config.EXPERIMENTS_DIR, "reports")
        os.makedirs(out_dir, exist_ok=True)
        props_json_formatted = json.dumps(props, indent=2)

        chart_section = ""
        if chart_paths:
            chart_section = "\n".join(
                f"![Chart {i+1}]({os.path.basename(c['path'])})\n\n*{c['caption']}*\n" for i, c in enumerate(chart_paths)
            )
        thermo_section = ""
        if thermo_paths:
            thermo_section = "\n".join(
                f"![Thermo Chart {i+1}]({os.path.basename(c['path'])})\n\n*{c['caption']}*\n" for i, c in enumerate(thermo_paths)
            )

        # Build Thermodynamic Insights section
        thermo_insights_section = ""
        if thermo_insights:
            ti = thermo_insights

            thermo_insights_section = "## 5a. Thermodynamic Insights (PyCalphad)\n\n"

            # ── Melting Range & Weldability ────────────────────────────────────
            melting_rows = []
            if 'Solidus_Temperature_C' in ti:
                melting_rows.append(f"| Solidus Temperature (onset of melting) | **{ti['Solidus_Temperature_C']:.0f} °C** | {ti.get('Solidus_Temperature_K', 'N/A')} K |")
            if 'Liquidus_Temperature_C' in ti:
                melting_rows.append(f"| Liquidus Temperature (fully liquid) | **{ti['Liquidus_Temperature_C']:.0f} °C** | {ti.get('Liquidus_Temperature_K', 'N/A')} K |")
            if 'Freezing_Range_K' in ti:
                fr = ti['Freezing_Range_K']
                if fr < 50:
                    fr_rating = "Narrow"
                elif fr < 150:
                    fr_rating = "Moderate"
                else:
                    fr_rating = "Wide"
                melting_rows.append(f"| Freezing Range (Delta-T) | **{fr:.0f} K** | {fr_rating} |")
            if 'Weldability' in ti:
                weld = ti['Weldability']
                melting_rows.append(f"| Weldability Assessment | **{weld.split('(')[0].strip()}** | {weld[weld.find('('):] if '(' in weld else ''} |")

            if melting_rows:
                thermo_insights_section += "### Melting Behavior and Weldability\n\n"
                thermo_insights_section += "| Property | Value | Detail |\n|:---|:---|:---|\n"
                thermo_insights_section += "\n".join(melting_rows) + "\n\n"

            # ── Phase Stability ───────────────────────────────────────────────
            phase_rows = []
            if ti.get('Stable_Phases_At_Service_Temp'):
                phases = ti['Stable_Phases_At_Service_Temp']
                ref_temp = ti.get('Service_Temp_Reference_K', '~700')
                for phase_name, frac in sorted(phases.items(), key=lambda x: -x[1]):
                    phase_rows.append(f"| {phase_name} | **{frac*100:.1f}%** |")

                thermo_insights_section += f"### Equilibrium Phase Distribution at Service Temperature (~{ref_temp} K)\n\n"
                thermo_insights_section += "| Phase | Mole Fraction |\n|:---|:---|\n"
                thermo_insights_section += "\n".join(phase_rows) + "\n\n"

            if ti.get('Harmful_Phase_Warnings'):
                thermo_insights_section += "> **Harmful Phase Warnings**\n>\n"
                for w in ti['Harmful_Phase_Warnings']:
                    thermo_insights_section += f"> - {w}\n"
                thermo_insights_section += "\n"

            # ── Phase Transitions & Service Limits ────────────────────────────
            transition_rows = []
            if 'BCC_FCC_Transition_C' in ti:
                transition_rows.append(f"| Ae3 Temperature (BCC-to-FCC transition) | **{ti['BCC_FCC_Transition_C']:.0f} °C** | Ferrite-to-austenite phase boundary |")
            if 'Max_Service_Temperature_C' in ti:
                phase = ti.get('Max_Service_Dominant_Phase', 'primary phase')
                transition_rows.append(f"| Maximum Service Temperature | **{ti['Max_Service_Temperature_C']:.0f} °C** | {phase} remains stable (>10 mol%) |")

            if transition_rows:
                thermo_insights_section += "### Critical Phase Transitions and Service Limits\n\n"
                thermo_insights_section += "| Property | Value | Note |\n|:---|:---|:---|\n"
                thermo_insights_section += "\n".join(transition_rows) + "\n\n"

            # If nothing was generated at all, show a note
            if not melting_rows and not phase_rows and not transition_rows:
                thermo_insights_section += "*No actionable thermodynamic insights could be extracted from the simulation data.*\n\n"

        # Build Iteration History section
        iteration_section = ""
        if iteration_history and len(iteration_history) > 1:
            iteration_section = "## 7. Iteration History\n\n"
            iteration_section += "| Iteration | Composition | UTS (MPa) | PREN | Elongation (%) | Failures |\n"
            iteration_section += "|---|---|---|---|---|---|\n"
            for ih in iteration_history:
                p = ih['props']
                fail_str = ', '.join(f['metric'] for f in ih['failures']) if ih['failures'] else 'All targets passed'
                iteration_section += (
                    f"| {ih['iteration']} | {ih['composition_str']} | "
                    f"{p.get('Estimated_UTS_MPa', 'N/A')} | "
                    f"{p.get('PREN_Corrosion_Index', 'N/A')} | "
                    f"{p.get('Estimated_Elongation_pct', 'N/A')} | "
                    f"{fail_str} |\n"
                )

        md_content = f"""# Autonomous Discovery Report: {report_id}

## 1. User Objective
**Query:** `{goal}`

## 2. Hypothesis Generation
* **Raw AI Output:** `{raw_ai}`
* **Parsed Composition:** `{comp}`

## 3. Physics Simulation Results
```json
{props_json_formatted}
```
## 4. Mechanical Analysis Charts
{chart_section}

{thermo_insights_section}

## 5b. Thermodynamic Phase & Energy Charts
{thermo_section}

## 6. Analyst Conclusion
{analysis}

{iteration_section}

Generated by Meta-Scientific Agent
"""
        md_path = os.path.join(out_dir, f"report_{report_id}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        return md_path

class MetaScientificAgent:
    def __init__(self):
        print(f"[*] Initializing Universal Framework with {config.ACTIVE_MODEL_NAME}...")

        # 1. Load RAG (CPU Offload)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        self.aisi_index = faiss.read_index(os.path.join(config.EMBEDDINGS_DIR, "academic_aisi.index"))
        self.rules_index = faiss.read_index(os.path.join(config.EMBEDDINGS_DIR, "metallurgical_reasoning.index"))

        with open(os.path.join(config.EMBEDDINGS_DIR, "academic_aisi_metadata.json"), "r") as f:
            self.aisi_meta = json.load(f)
        with open(os.path.join(config.EMBEDDINGS_DIR, "metallurgical_reasoning_metadata.json"), "r") as f:
            self.rules_meta = json.load(f)
            
        try:
            self.apps_index = faiss.read_index(os.path.join(config.EMBEDDINGS_DIR, "industrial_applications.index"))
            with open(os.path.join(config.EMBEDDINGS_DIR, "industrial_applications_metadata.json"), "r") as f:
                self.apps_meta = json.load(f)
        except Exception as e:
            print(f"[-] Could not load applications index: {e}")
            self.apps_index = None
            self.apps_meta = None

        # 2. Load Base Model Configured for VRAM limits
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="cuda:0",
            local_files_only=True
        )

        # 3. Load LoRA Adapters
        peft_model = PeftModel.from_pretrained(base_model, config.HYPOTHESIS_ADAPTER_DIR, adapter_name="hypothesis")
        peft_model.load_adapter(config.ANALYST_ADAPTER_DIR, adapter_name="analyst")

        # 4. Initialize Custom LangChain LLM
        self.llm = LoRASwitchingLLM(model=peft_model, tokenizer=tokenizer)

        # 5. Define LangChain Prompts
        # NOTE: this template must match generate_taining_samples.py's prompt
        # EXACTLY (same instruction text, same section structure, no extra
        # few-shot block) -- the hypothesis adapter was fine-tuned on that
        # exact format and nothing else. A previous version of this template
        # inserted an additional "--- Example ---" few-shot block before
        # "--- Actual Task ---" that the adapter never saw during training;
        # this caused occasional degenerate generations where the model
        # hallucinated a second, spurious section (mimicking the unfamiliar
        # two-block structure) instead of directly completing the composition,
        # sometimes omitting Fe entirely. Removed to restore train/inference
        # alignment.
        self.hypo_prompt = PromptTemplate(
            input_variables=["goal", "baseline", "rule", "app_context"],
            template=(
                "Task: Propose a new alloy chemistry based on the baseline.\n"
                "Output ONLY the elements and percentages. Do not write anything else.\n"
                "Format: Element-Percentage Element-Percentage\n\n"
                "--- Actual Task ---\n"
                "Goal: {goal}\n"
                "Application Context: {app_context}\n"
                "Baseline: {baseline}\n"
                "Rule: {rule}\n"
                "Proposed Chemistry: "
            )
        )
        
        self.analyst_prompt = PromptTemplate(
            input_variables=["goal", "results"],
            template=(
                "You are an expert metallurgical analyst.\n"
                "Goal: {goal}\n"
                "Simulation Results: {results}\n"
                "Conclusion (maximum 2 sentences): "
            )
        )

        # 6. Build LCEL Chains
        self.hypo_chain = self.hypo_prompt | self.llm | StrOutputParser()
        self.analyst_chain = self.analyst_prompt | self.llm | StrOutputParser()

        print("[+] LangChain Orchestration Ready.")

    def run_discovery(self, goal):
        try:
            report_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # --- RAG Retrieval ---
            print(f"\n[*] Querying FAISS embeddings for: '{goal}'...")
            query_vec = self.embed_model.encode([goal]).astype('float32')
            _, aisi_idx = self.aisi_index.search(query_vec, 1)
            _, rules_idx = self.rules_index.search(query_vec, 1)
            
            baseline = self.aisi_meta[aisi_idx[0][0]]
            rule = self.rules_meta[rules_idx[0][0]]
            
            app_context = "None"
            if self.apps_index:
                _, apps_idx = self.apps_index.search(query_vec, 1)
                app_context = self.apps_meta[apps_idx[0][0]]
                print(f"[*] Retrieved Application: {app_context}")
            
            print(f"[*] Retrieved Baseline: {baseline}")
            print(f"[*] Retrieved Rule: {rule}")

            # --- Phase 1: Hypothesis Generation ---
            print("\n[*] Invoking LLM (Hypothesis Adapter)...")
            self.llm.set_adapter("hypothesis")
            raw_ai = self.hypo_chain.invoke({
                "goal": goal, 
                "baseline": baseline, 
                "rule": rule,
                "app_context": app_context
            })
            
            print(f"\n[AI HYPOTHESIS OUTPUT]:\n{raw_ai}\n")

            comp = ScientificEngine.parse_composition(raw_ai)
            if not comp:
                print("[-] Error: AI failed to propose valid stoichiometry.")
                return {"error": "AI failed to propose valid stoichiometry."}

            print(f"[+] Parsed Composition: {comp}")

            # --- Stoichiometric Plausibility Guard ---
            # The framework's scope is restricted to Fe-based industrial steel
            # alloys. Reject compositions with insufficient Fe content instead
            # of silently computing property estimates for degenerate,
            # out-of-scope generations (e.g. near-pure-carbon or Ni-majority
            # outputs) -- such compositions previously produced physically
            # impossible property estimates (UTS in excess of 10 GPa, versus
            # a realistic ceiling around 2-3 GPa for any known alloy) that were
            # then compared against real industry-standard targets.
            if not ScientificEngine.is_plausible_composition(comp):
                fe_pct = comp.get("Fe", 0.0) * 100
                print(f"[-] Constraint Failure: composition is not Fe-based (Fe={fe_pct:.1f}%), outside framework scope.")
                return {
                    "error": "Constraint Failure: Alloy must be Iron-based (>30% Fe) to be within framework scope.",
                    "Parsed_Composition": comp,
                }

            # --- Physics Validation ---
            print("\n[*] Running Physics Validation...")
            props = ScientificEngine.calculate_physics(comp, heat_treatment="annealed")
            
            # --- Parse Industry Standard and Compute Comparison ---
            if app_context != "None":
                try:
                    import re
                    # Extract Alloy Name
                    alloy_match = re.search(r'\| alloy \|([^\]]+)\]', app_context)
                    if alloy_match:
                        props["Industry_Standard_Alloy"] = alloy_match.group(1).strip()
                        
                    # Extract UTS
                    uts_match = re.search(r'\| UTS \| ([0-9.]+) MPa', app_context)
                    if uts_match:
                        std_uts = float(uts_match.group(1))
                        props["Industry_Standard_UTS_MPa"] = std_uts
                        diff_uts = round(props["Estimated_UTS_MPa"] - std_uts, 1)
                        if diff_uts >= 0:
                            props["Comparison_UTS"] = f"PASS (+{diff_uts} MPa over standard)"
                        else:
                            props["Comparison_UTS"] = f"FAIL ({diff_uts} MPa below standard)"
                except Exception as e:
                    print(f"[-] Could not parse industry standard for comparison: {e}")
                    
            print(f"[+] Calculated Properties & Comparisons: {props}")

            # --- Phase 2: Analyst Evaluation ---
            print("\n[*] Invoking LLM (Analyst Adapter)...")
            self.llm.set_adapter("analyst")
            analysis_text = self.analyst_chain.invoke({
                "goal": goal,
                "results": json.dumps(props)
            })
            
            print(f"\n[AI ANALYST OUTPUT]:\n{analysis_text}\n")

            # --- Calphad Phase Diagram Generation ---
            print("\n[*] Running PyCalphad Phase Diagram Calculation...")
            tdb_file = os.path.join(config.TDB_DIR, "mc_fe_pycalphad.tdb")
            phase_data = ScientificEngine.generate_phase_data(comp, tdb_file)

            # Create run directory
            run_dir = os.path.join(config.EXPERIMENTS_DIR, "runs", report_id)
            os.makedirs(run_dir, exist_ok=True)
            
            # --- Reporting ---
            print(f"[*] Plotting results for {comp}...")
            chart_paths  = DiscoveryReporter.generate_dashboard(props, comp, report_id, output_dir=run_dir)
            thermo_paths = DiscoveryReporter.generate_thermo_dashboard(phase_data, report_id, output_dir=run_dir)

            print("[*] Generating Markdown report...")
            md_path = DiscoveryReporter.generate_markdown_report(
                goal=goal, raw_ai=raw_ai, comp=comp, props=props,
                analysis=analysis_text, report_id=report_id,
                chart_paths=chart_paths, thermo_paths=thermo_paths, output_dir=run_dir
            )

            # Append to self-evolution queue
            queue_path = os.path.join(config.EXPERIMENTS_DIR, "self_evolution_queue.jsonl")
            with open(queue_path, "a") as f:
                f.write(json.dumps({"goal": goal, "hypothesis": raw_ai, "analysis": analysis_text}) + "\n")

            print(f"[+] Discovery Report Generated: {md_path}")
            
            return {
                "report_id": report_id,
                "md_path": md_path,
                "chart_paths": chart_paths,
                "thermo_paths": thermo_paths,
                "goal": goal,
                "hypothesis": raw_ai,
                "analysis": analysis_text
            }

        except Exception as e:
            print(f"[-] Discovery Failed: {e}")
            return {"error": str(e)}
if __name__ == "__main__":
    agent = MetaScientificAgent()

    while True:
        user_query = input("\n[?] Enter discovery objective (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
        
        agent.run_discovery(user_query)
        torch.cuda.empty_cache()