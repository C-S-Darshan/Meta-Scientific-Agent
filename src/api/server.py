import os
import time
import sys
import shutil
import glob
import re
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add src to Python path so we can import config and agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Trigger reload again to pick up the very final agent.py changes
# Trigger reload again to pick up the final agent.py changes
# Trigger reload


from agent import MetaScientificAgent
import config

app = FastAPI(title="Meta-Scientific Agent API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for runs
runs_dir = os.path.join(config.EXPERIMENTS_DIR, "runs")
os.makedirs(runs_dir, exist_ok=True)
app.mount("/api/runs", StaticFiles(directory=runs_dir), name="runs")

# Initialize agent globally
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    print("[API] Initializing Meta-Scientific Agent...")
    agent = MetaScientificAgent()
    print("[API] Agent initialized and ready.")

class PromptRequest(BaseModel):
    prompt: str
    max_iterations: int = 1

def _evaluate_against_targets(props, app_context, targets_override=None):
    """
    Compare physics props against industry targets extracted from app_context.
    Returns (props_updated, failures_list, targets_dict).
    """
    failures = []
    targets = targets_override if targets_override else {"Pugh_Ratio": 1.75}
    if app_context == "None":
        return props, failures, targets
    try:
        alloy_match = re.search(r'\| alloy \|([^\]]+)\]', app_context)
        if alloy_match:
            props["Industry_Standard_Alloy"] = alloy_match.group(1).strip()

        uts_match = re.search(r'\| UTS \| ([0-9.]+) MPa', app_context)
        if uts_match:
            std_uts = float(uts_match.group(1))
            props["Industry_Standard_UTS_MPa"] = std_uts
            targets["UTS"] = std_uts
            diff_uts = round(props["Estimated_UTS_MPa"] - std_uts, 1)
            if diff_uts >= 0:
                props["Comparison_UTS"] = f"PASS (+{diff_uts} MPa over standard)"
            else:
                props["Comparison_UTS"] = f"FAIL ({diff_uts} MPa below standard)"
                failures.append({"metric": "UTS", "actual": props["Estimated_UTS_MPa"], "target": std_uts, "gap": diff_uts})

        pren_match = re.search(r'\| PREN \| ([0-9.]+)', app_context)
        if pren_match:
            std_pren = float(pren_match.group(1))
            props["Industry_Standard_PREN"] = std_pren
            targets["PREN"] = std_pren
            diff_pren = round(props.get("PREN_Corrosion_Index", 0) - std_pren, 2)
            if diff_pren >= 0:
                props["Comparison_PREN"] = f"PASS (+{diff_pren} over standard)"
            else:
                props["Comparison_PREN"] = f"FAIL ({diff_pren} below standard)"
                failures.append({"metric": "PREN", "actual": props.get("PREN_Corrosion_Index", 0), "target": std_pren, "gap": diff_pren})

        elong_match = re.search(r'\| elongation \| ([0-9.]+)%', app_context)
        if elong_match:
            std_elong = float(elong_match.group(1))
            props["Industry_Standard_Elongation"] = std_elong
            targets["Elongation"] = std_elong
            diff_elong = round(props.get("Estimated_Elongation_pct", 0) - std_elong, 2)
            if diff_elong >= 0:
                props["Comparison_Elongation"] = f"PASS (+{diff_elong}% over standard)"
            else:
                props["Comparison_Elongation"] = f"FAIL ({diff_elong}% below standard)"
                failures.append({"metric": "Elongation", "actual": props.get("Estimated_Elongation_pct", 0), "target": std_elong, "gap": diff_elong})
    except Exception:
        pass
    return props, failures, targets


def _apply_corrections(comp, failures):
    """Apply targeted element adjustments to a composition based on failures."""
    import copy
    corrected = copy.deepcopy(comp)
    adjustment_notes = []
    for fail in failures:
        if fail["metric"] == "PREN":
            corrected["Cr"] = corrected.get("Cr", 0) + 0.03
            corrected["Mo"] = corrected.get("Mo", 0) + 0.015
            adjustment_notes.append(f"Increase Cr/Mo to raise PREN from {fail['actual']:.1f} toward {fail['target']:.1f}")
        elif fail["metric"] == "UTS":
            corrected["C"]  = corrected.get("C", 0)  + 0.002
            corrected["V"]  = corrected.get("V", 0)  + 0.005
            corrected["Ni"] = corrected.get("Ni", 0) + 0.02
            adjustment_notes.append(f"Increase C/V/Ni to raise UTS from {fail['actual']:.0f} toward {fail['target']:.0f} MPa")
        elif fail["metric"] == "Elongation":
            corrected["Ni"] = corrected.get("Ni", 0) + 0.03
            adjustment_notes.append(f"Increase Ni to raise elongation from {fail['actual']:.1f} toward {fail['target']:.1f}%")
    # Re-normalise
    total_w = sum(corrected.values())
    if total_w > 0:
        corrected = {k: round(v / total_w, 4) for k, v in corrected.items()}
    return corrected, adjustment_notes


def run_discovery_with_steps(prompt: str, max_iterations: int = 1):
    """
    Generator that runs the discovery pipeline step by step,
    yielding SSE-formatted progress events at each stage.
    Supports iterative retry on failure when max_iterations > 1.
    """
    import formulas as ScientificEngine
    from datetime import datetime

    def emit(event_type: str, payload: dict):
        data = json.dumps({"type": event_type, **payload})
        return f"data: {data}\n\n"

    try:
        report_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Step 1: Encoding query ────────────────────────────────────────
        t0_embed = time.time()
        yield emit("step", {"id": "embed", "status": "active",
                             "label": "Encoding your query into semantic vectors..."})
        query_vec = agent.embed_model.encode([prompt]).astype('float32')
        yield emit("step", {"id": "embed", "status": "done",
                             "label": "Query encoded into semantic vectors", "duration": f"{time.time() - t0_embed:.1f}s"})

        # ── Step 2: Fetching baseline alloy ───────────────────────────────
        t0_rag_alloy = time.time()
        yield emit("step", {"id": "rag_alloy", "status": "active",
                             "label": "Searching alloy knowledge base for closest baseline..."})
        _, aisi_idx = agent.aisi_index.search(query_vec, 1)
        baseline = agent.aisi_meta[aisi_idx[0][0]]
        alloy_id = baseline.split(" | ")[0].strip("[") if "[" in baseline else "Alloy"
        yield emit("step", {"id": "rag_alloy", "status": "done",
                             "label": f"Baseline retrieved: {alloy_id}",
                             "detail": baseline, "duration": f"{time.time() - t0_rag_alloy:.1f}s"})

        # ── Step 3: Fetching metallurgical rule ───────────────────────────
        t0_rag_rule = time.time()
        yield emit("step", {"id": "rag_rule", "status": "active",
                             "label": "Retrieving best-matching metallurgical rule..."})
        _, rules_idx = agent.rules_index.search(query_vec, 1)
        rule = agent.rules_meta[rules_idx[0][0]]
        rule_parts = rule.strip("[]").split(" | ")
        element = rule_parts[0] if len(rule_parts) > 0 else "element"
        goal_str = rule_parts[1].replace("goal: ", "") if len(rule_parts) > 1 else ""
        yield emit("step", {"id": "rag_rule", "status": "done",
                             "label": f"Rule matched: {element} → {goal_str}",
                             "detail": rule, "duration": f"{time.time() - t0_rag_rule:.1f}s"})

        # ── Step 4: Fetching application context (optional) ───────────────
        app_context = "None"
        if agent.apps_index:
            t0_rag_app = time.time()
            yield emit("step", {"id": "rag_app", "status": "active",
                                 "label": "Matching industrial application context..."})
            _, apps_idx = agent.apps_index.search(query_vec, 1)
            app_context = agent.apps_meta[apps_idx[0][0]]
            app_label = "General"
            for part in app_context.split("], ["):
                if "application |" in part.lower():
                    app_label = part.split("| application |")[-1].strip().strip("[]")
                    break
            yield emit("step", {"id": "rag_app", "status": "done",
                                 "label": f"Application context: {app_label}",
                                 "detail": app_context, "duration": f"{time.time() - t0_rag_app:.1f}s"})

        # ── Step 5: Hypothesis generation ─────────────────────────────────
        t0_hypothesis = time.time()
        yield emit("step", {"id": "hypothesis", "status": "active",
                             "label": "Activating Hypothesis LoRA adapter — generating alloy composition..."})
        agent.llm.set_adapter("hypothesis")
        raw_ai = agent.hypo_chain.invoke({
            "goal": prompt,
            "baseline": baseline,
            "rule": rule,
            "app_context": app_context
        })
        comp = ScientificEngine.parse_composition(raw_ai)
        if not comp:
            yield emit("error", {"message": "AI failed to propose a valid stoichiometry."})
            return
        comp_str = ", ".join(f"{k}-{round(v*100,1)}%" for k, v in comp.items())
        yield emit("step", {"id": "hypothesis", "status": "done",
                             "label": f"Hypothesis generated: {comp_str}", "duration": f"{time.time() - t0_hypothesis:.1f}s"})

        # ══════════════════════════════════════════════════════════════════
        # ── ITERATIVE REFINEMENT LOOP ─────────────────────────────────────
        # ══════════════════════════════════════════════════════════════════
        iteration_history = []  # Track all iterations for the report

        for iteration in range(1, max_iterations + 1):
            is_retry = iteration > 1
            iter_prefix = f"[Iter {iteration}/{max_iterations}] " if max_iterations > 1 else ""

            if is_retry:
                yield emit("step", {"id": f"iter_{iteration}", "status": "active",
                                     "label": f"Iteration {iteration}/{max_iterations} — refining composition with targeted adjustments..."})

            # ── Step 6: Physics engine ─────────────────────────────────────
            t0_physics = time.time()
            yield emit("step", {"id": f"physics_{iteration}" if is_retry else "physics", "status": "active",
                                 "label": f"{iter_prefix}Running physics engine — estimating UTS, PREN, density..."})
            props = ScientificEngine.calculate_physics(comp, heat_treatment="annealed")

            # --- Evaluate against targets ---
            props, failures, targets = _evaluate_against_targets(props, app_context)

            uts = props.get('Estimated_UTS_MPa', 'N/A')
            pren = props.get('PREN_Corrosion_Index', 'N/A')
            yield emit("step", {"id": f"physics_{iteration}" if is_retry else "physics", "status": "done",
                                 "label": f"{iter_prefix}Physics validated — UTS: {uts} MPa | PREN: {pren}", "duration": f"{time.time() - t0_physics:.1f}s"})

            # ── Step 6b: Heat Treatment Recommendation ─────────────────────
            uts_or_elong_failures = [f for f in failures if f["metric"] in ("UTS", "Elongation")]
            if uts_or_elong_failures and targets:
                yield emit("step", {"id": f"heat_treat_{iteration}" if is_retry else "heat_treat", "status": "active",
                                     "label": f"{iter_prefix}Scanning heat treatments to meet targets..."})
                ht_rec = ScientificEngine.recommend_heat_treatment(comp, targets)
                rec_name = ht_rec["recommended_treatment"]

                detail_lines = []
                for tr in ht_rec["all_treatments"]:
                    status = "[PASS]" if tr["all_pass"] else "[FAIL]"
                    detail_lines.append(f"{status} {tr['treatment']}: UTS={tr['UTS_MPa']} MPa | Elong={tr['Elongation_pct']}%")
                detail_str = "\n".join(detail_lines)

                if ht_rec["meets_all_targets"]:
                    props = ScientificEngine.calculate_physics(comp, heat_treatment=rec_name)
                    props, _, _ = _evaluate_against_targets(props, app_context, targets)
                    props["Recommended_Heat_Treatment"] = rec_name
                    failures = [f for f in failures if f["metric"] not in ("UTS", "Elongation")]
                    yield emit("step", {"id": f"heat_treat_{iteration}" if is_retry else "heat_treat", "status": "done",
                                         "label": f"{iter_prefix}[PASS] '{rec_name}' meets targets — UTS: {ht_rec['recommended_UTS_MPa']} MPa",
                                         "detail": detail_str})
                else:
                    props["Recommended_Heat_Treatment"] = f"{rec_name} (closest, still fails)"
                    yield emit("step", {"id": f"heat_treat_{iteration}" if is_retry else "heat_treat", "status": "done",
                                         "label": f"{iter_prefix}[CAUTION] No treatment fully meets targets — closest: '{rec_name}'",
                                         "detail": detail_str})

            # Record this iteration
            comp_str_iter = ", ".join(f"{k}-{round(v*100,1)}%" for k, v in comp.items())
            iteration_history.append({
                "iteration": iteration,
                "composition": dict(comp),
                "composition_str": comp_str_iter,
                "props": dict(props),
                "failures": list(failures),
            })

            # Check if we should retry
            if failures and iteration < max_iterations:
                corrected, adj_notes = _apply_corrections(comp, failures)
                failed_metrics = ", ".join(f["metric"] for f in failures)
                adj_detail = "\n".join(adj_notes)
                corrected_str = ", ".join(f"{k}-{round(v*100,1)}%" for k, v in corrected.items())

                yield emit("step", {"id": f"iter_{iteration}", "status": "done",
                                     "label": f"Iteration {iteration} failed ({failed_metrics}) — adjusting composition for retry",
                                     "detail": f"Adjustments:\n{adj_detail}\n\nNew composition: {corrected_str}"})
                comp = corrected
                continue  # Go to next iteration
            else:
                if is_retry:
                    if not failures:
                        yield emit("step", {"id": f"iter_{iteration}", "status": "done",
                                             "label": f"[PASS] Iteration {iteration} — all targets met!"})
                    else:
                        failed_metrics = ", ".join(f["metric"] for f in failures)
                        yield emit("step", {"id": f"iter_{iteration}", "status": "done",
                                             "label": f"[CAUTION] Iteration {iteration}/{max_iterations} — still failing: {failed_metrics} (max iterations reached)"})
                break  # All passed or max iterations reached

        # ══════════════════════════════════════════════════════════════════
        # ── POST-LOOP: Analyst, Calphad, Reporting ────────────────────────
        # ══════════════════════════════════════════════════════════════════

        # ── Step 8: PyCalphad simulation ───────────────────────────────────
        t0_calphad = time.time()
        yield emit("step", {"id": "calphad", "status": "active",
                             "label": "Running PyCalphad thermodynamic phase simulation..."})
        tdb_file = os.path.join(config.TDB_DIR, "mc_fe_pycalphad.tdb")
        phase_data = ScientificEngine.generate_phase_data(comp, tdb_file)
        calphad_label = "Thermodynamic phase diagram computed" if phase_data else "Calphad simulation skipped (incompatible TDB)"
        yield emit("step", {"id": "calphad", "status": "done", "label": calphad_label, "duration": f"{time.time() - t0_calphad:.1f}s"})

        # ── Step 8b: Extract Thermodynamic Insights ───────────────────────
        thermo_insights = None
        if phase_data:
            yield emit("step", {"id": "thermo_insights", "status": "active",
                                 "label": "Extracting thermodynamic insights (solidus, liquidus, phase stability)..."})
            thermo_insights = ScientificEngine.extract_thermo_insights(phase_data, comp)
            if thermo_insights:
                # Merge key insights into props for the report
                for key in ['Solidus_Temperature_C', 'Liquidus_Temperature_C', 'Freezing_Range_K',
                            'Weldability', 'BCC_FCC_Transition_C', 'Max_Service_Temperature_C']:
                    if key in thermo_insights:
                        props[f"Thermo_{key}"] = thermo_insights[key]
                if thermo_insights.get('Harmful_Phase_Warnings'):
                    props["Thermo_Harmful_Phase_Warnings"] = thermo_insights['Harmful_Phase_Warnings']

                insight_label = []
                if 'Solidus_Temperature_C' in thermo_insights:
                    insight_label.append(f"Solidus: {thermo_insights['Solidus_Temperature_C']:.0f}°C")
                if 'Weldability' in thermo_insights:
                    weld_short = thermo_insights['Weldability'].split('(')[0].strip()
                    insight_label.append(f"Weldability: {weld_short}")
                yield emit("step", {"id": "thermo_insights", "status": "done",
                                     "label": f"Thermo insights extracted — {' | '.join(insight_label) if insight_label else 'data processed'}"})
            else:
                yield emit("step", {"id": "thermo_insights", "status": "done",
                                     "label": "No actionable thermodynamic insights available"})

        # ── Step 7: Analyst evaluation (with thermo insights) ─────────────
        t0_analyst = time.time()
        yield emit("step", {"id": "analyst", "status": "active",
                             "label": "Activating Analyst LoRA adapter — evaluating results..."})
        agent.llm.set_adapter("analyst")
        from langchain_core.output_parsers import StrOutputParser

        # Build enriched results string including thermo insights
        analyst_results = json.dumps(props)
        if thermo_insights and thermo_insights.get('_analyst_summary'):
            analyst_results += f"\nThermodynamic Analysis: {thermo_insights['_analyst_summary']}"

        analysis_text = agent.analyst_chain.invoke({
            "goal": prompt,
            "results": analyst_results
        })
        yield emit("step", {"id": "analyst", "status": "done",
                             "label": "Analyst conclusion ready", "duration": f"{time.time() - t0_analyst:.1f}s"})

        # Create run directory
        run_dir = os.path.join(config.EXPERIMENTS_DIR, "runs", report_id)
        os.makedirs(run_dir, exist_ok=True)

        # ── Step 9: Generating dashboards ──────────────────────────────────────
        t0_report = time.time()
        yield emit("step", {"id": "report", "status": "active",
                             "label": "Generating visual dashboards and discovery report..."})
        from agent import DiscoveryReporter
        chart_paths  = DiscoveryReporter.generate_dashboard(props, comp, report_id, output_dir=run_dir, targets=targets)
        thermo_paths = DiscoveryReporter.generate_thermo_dashboard(phase_data, report_id, output_dir=run_dir)
        md_path = DiscoveryReporter.generate_markdown_report(
            goal=prompt, raw_ai=raw_ai, comp=comp, props=props,
            analysis=analysis_text, report_id=report_id,
            chart_paths=chart_paths, thermo_paths=thermo_paths, output_dir=run_dir,
            thermo_insights=thermo_insights, iteration_history=iteration_history
        )

        # Append to self-evolution queue
        queue_path = os.path.join(config.EXPERIMENTS_DIR, "self_evolution_queue.jsonl")
        with open(queue_path, "a") as f:
            f.write(json.dumps({"goal": prompt, "hypothesis": raw_ai, "analysis": analysis_text}) + "\n")

        # ── Persist RAG context for history viewing ────────────────────
        rag_context = {
            "baseline": baseline,
            "rule": rule,
            "app_context": app_context,
        }
        context_path = os.path.join(run_dir, "context.json")
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(rag_context, f, indent=2)

        yield emit("step", {"id": "report", "status": "done",
                             "label": f"{len(chart_paths) + len(thermo_paths)} individual charts saved", "duration": f"{time.time() - t0_report:.1f}s"})

        # ── Step 10: Self-Evolution on Failure ──────────────────────────────
        if failures:
            yield emit("step", {"id": "evolution", "status": "active",
                                 "label": f"Alloy FAILED {len(failures)} target(s) — generating corrective training samples..."})
            evo_path = os.path.join(config.EXPERIMENTS_DIR, "self_evolution_corrections.jsonl")

            corrected, adjustment_notes = _apply_corrections(comp, failures)
            corrected_str = " ".join(f"{k}-{round(v*100, 1)}" for k, v in corrected.items())
            failed_metrics = ", ".join(f["metric"] for f in failures)

            hypo_text = (
                f"Task: Propose a new alloy chemistry based on the baseline.\n"
                f"Output ONLY the elements and percentages. Do not write anything else.\n"
                f"Format: Element-Percentage Element-Percentage\n\n"
                f"--- Actual Task ---\n"
                f"Goal: {prompt}\n"
                f"Application Context: {app_context}\n"
                f"Baseline: {baseline}\n"
                f"Rule: {rule}\n"
                f"Proposed Chemistry: {corrected_str}"
            )

            analyst_text_correction = (
                f"You are an expert metallurgical analyst.\n"
                f"Goal: {prompt}\n"
                f"Simulation Results: {json.dumps(props)}\n"
                f"Conclusion (maximum 2 sentences): "
                f"The proposed alloy fails to meet the industry standard for {failed_metrics}. "
                f"Recommended adjustments: {'; '.join(adjustment_notes)}."
            )

            num_corrections = 0
            with open(evo_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "hypothesis", "text": hypo_text, "failures": [fl["metric"] for fl in failures]}) + "\n")
                f.write(json.dumps({"type": "analyst",    "text": analyst_text_correction, "failures": [fl["metric"] for fl in failures]}) + "\n")
                num_corrections = 2

            yield emit("step", {"id": "evolution", "status": "done",
                                 "label": f"Self-evolution: {num_corrections} corrective samples generated (failed: {failed_metrics})",
                                 "detail": "\n".join(adjustment_notes)})
        else:
            yield emit("step", {"id": "evolution", "status": "done",
                                 "label": "All targets met — no corrective training needed"})

        # ── Final result ───────────────────────────────────────────────────
        md_content = ""
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()

        yield emit("result", {
            "report_id": report_id,
            "markdown": md_content,
            "chart_urls":  [{"url": f"/api/runs/{report_id}/{os.path.basename(p['path'])}", "caption": p['caption']} for p in chart_paths],
            "thermo_urls": [{"url": f"/api/runs/{report_id}/{os.path.basename(p['path'])}", "caption": p['caption']} for p in thermo_paths],
            "goal": prompt
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield emit("error", {"message": str(e)})


@app.post("/api/chat/stream")
async def chat_stream(request: PromptRequest):
    """SSE endpoint — streams progress steps then delivers the final result."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent is not initialized yet.")

    max_iter = max(1, min(request.max_iterations, 5))  # Clamp to 1-5

    def generate():
        for chunk in run_discovery_with_steps(request.prompt, max_iterations=max_iter):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/chat")
async def chat(request: PromptRequest):
    """Non-streaming fallback endpoint (kept for compatibility)."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent is not initialized yet.")

    print(f"[API] Received prompt: {request.prompt}")
    try:
        result = agent.run_discovery(request.prompt)
        if result and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        if not result:
            raise HTTPException(status_code=500, detail="Agent returned empty result.")

        md_content = ""
        if os.path.exists(result["md_path"]):
            with open(result["md_path"], "r", encoding="utf-8") as f:
                md_content = f.read()

        chart_paths  = result.get("chart_paths", []) or []
        thermo_paths = result.get("thermo_paths", []) or []

        return {
            "report_id": result["report_id"],
            "markdown": md_content,
            "chart_urls":  [{"url": f"/api/runs/{result['report_id']}/{os.path.basename(p['path'])}", "caption": p['caption']} for p in chart_paths],
            "thermo_urls": [{"url": f"/api/runs/{result['report_id']}/{os.path.basename(p['path'])}", "caption": p['caption']} for p in thermo_paths],
            "goal": result["goal"]
        }
    except Exception as e:
        print(f"[API] Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history():
    runs_dir = os.path.join(config.EXPERIMENTS_DIR, "runs")
    if not os.path.exists(runs_dir):
        return {"history": []}

    history = []
    for run_folder in glob.glob(os.path.join(runs_dir, "*")):
        if not os.path.isdir(run_folder):
            continue
        report_id = os.path.basename(run_folder)
        md_path = os.path.join(run_folder, f"report_{report_id}.md")
        if not os.path.exists(md_path):
            continue

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        goal_match = re.search(r"\*\*Query:\*\*\s*`(.*?)`", content)
        goal = goal_match.group(1) if goal_match else "Unknown Query"

        chart_urls = []
        thermo_urls = []
        for p in glob.glob(os.path.join(run_folder, "chart_*.png")):
            basename = os.path.basename(p)
            url = f"/api/runs/{report_id}/{basename}"
            
            # Simple caption extraction from markdown
            caption_match = re.search(r"!\[.*?\]\(" + re.escape(basename) + r"\)\n\n\*(.*?)\*", content, re.DOTALL)
            caption = caption_match.group(1).replace('\n', ' ') if caption_match else "View graph for details"

            item = {"url": url, "caption": caption}
            if "thermo" in basename or "phase" in basename or "heat" in basename or "enthalpy" in basename or "chem" in basename:
                thermo_urls.append(item)
            else:
                chart_urls.append(item)

        # Load persisted RAG context if available
        rag_context = None
        context_path = os.path.join(run_folder, "context.json")
        if os.path.exists(context_path):
            try:
                with open(context_path, "r", encoding="utf-8") as cf:
                    rag_context = json.load(cf)
            except Exception:
                pass

        history.append({
            "report_id": report_id,
            "goal": goal,
            "markdown": content,
            "chart_urls": chart_urls,
            "thermo_urls": thermo_urls,
            "rag_context": rag_context,
            "timestamp": os.path.getmtime(md_path)
        })

    history.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"history": history}


@app.delete("/api/history/{report_id}")
async def delete_history(report_id: str):
    """Delete a specific report and all its associated files."""
    run_dir = os.path.join(config.EXPERIMENTS_DIR, "runs", report_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="Report not found.")
    try:
        shutil.rmtree(run_dir)
        return {"message": "Report deleted successfully.", "report_id": report_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
