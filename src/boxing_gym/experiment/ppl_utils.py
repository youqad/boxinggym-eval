import re
from dataclasses import dataclass, field


@dataclass
class PPLArtifacts:
    only_programs: list = field(default_factory=list)
    programs_all_text: list = field(default_factory=list)
    program_entries: list = field(default_factory=list)
    best_plot_candidates: list = field(default_factory=list)
    round_stats_entries: list = field(default_factory=list)


def extract_ppl_artifacts(all_data, use_ppl: bool) -> PPLArtifacts:
    artifacts = PPLArtifacts()
    if not use_ppl or len(all_data) <= 6:
        return artifacts

    def _extract_code_blocks(text: str):
        if not text:
            return []
        return re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)

    def _first_code_block(text: str):
        blocks = _extract_code_blocks(text)
        return blocks[0].strip() if blocks else ""

    for round_idx, elem in enumerate(all_data[6]):
        round_programs = elem if isinstance(elem, list) else [elem]
        for prog_idx, prog in enumerate(round_programs):
            if isinstance(prog, dict):
                llm_text = prog.get("full_llm_response") or ""
                program_code = prog.get("str_prob_prog") or _first_code_block(llm_text)
                if llm_text:
                    artifacts.programs_all_text.append(llm_text)
                    if prog_idx == 0:
                        artifacts.only_programs.append(llm_text)

                artifacts.program_entries.append(
                    {
                        "round": round_idx,
                        "program_idx": prog_idx,
                        "program_code": program_code,
                        "loo": prog.get("loo"),
                        "waic": prog.get("waic"),
                        "summary_stats": prog.get("summary_stats"),
                        "llm_response": llm_text,
                        "n_divergences": prog.get("n_divergences"),
                        "max_rhat": prog.get("max_rhat"),
                        "min_ess_bulk": prog.get("min_ess_bulk"),
                        "diagnostics": prog.get("diagnostics"),
                    }
                )

                trace_obj = prog.get("trace")
                if trace_obj is not None:
                    loo_val = None
                    try:
                        loo_val = float(prog.get("loo"))
                    except Exception:
                        loo_val = None
                    artifacts.best_plot_candidates.append(
                        (loo_val, trace_obj, round_idx, prog_idx, program_code, prog)
                    )
            elif prog:
                llm_text = str(prog)
                program_code = _first_code_block(llm_text)
                artifacts.programs_all_text.append(llm_text)
                if prog_idx == 0:
                    artifacts.only_programs.append(llm_text)
                artifacts.program_entries.append(
                    {
                        "round": round_idx,
                        "program_idx": prog_idx,
                        "program_code": program_code,
                        "loo": "",
                        "waic": "",
                        "summary_stats": "",
                        "llm_response": llm_text,
                        "n_divergences": "",
                        "max_rhat": "",
                        "min_ess_bulk": "",
                        "diagnostics": "",
                    }
                )

    for entry in all_data[0]:
        try:
            if isinstance(entry, (list, tuple)) and len(entry) > 4 and isinstance(entry[4], list):
                artifacts.round_stats_entries.extend(entry[4])
        except Exception:
            continue

    return artifacts
