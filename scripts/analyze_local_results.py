#!/usr/bin/env python3
"""
Analyze local JSON results to find extreme predictions and extract raw LLM responses.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def find_extreme_predictions(results_dir: Path, threshold: float = 1e6) -> list[dict[str, Any]]:
    """Find results with extreme predictions."""
    extreme_cases = []

    # Recursively find all JSON files
    json_files = list(results_dir.rglob("*.json"))
    print(f"📂 Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract predictions from various possible structures
            predictions = []

            # Check step_results
            if "step_results" in data:
                for step in data["step_results"]:
                    if "prediction" in step:
                        pred = step["prediction"]
                        if isinstance(pred, (int, float)):
                            predictions.append(
                                {
                                    "value": pred,
                                    "step": step.get("step", "?"),
                                    "question": step.get("question", ""),
                                    "response": step.get("response", ""),
                                    "raw_response": step.get("raw_response", ""),
                                }
                            )

            # Check final_results
            if "final_results" in data:
                final = data["final_results"]
                if "predictions" in final:
                    for i, pred in enumerate(final["predictions"]):
                        if isinstance(pred, (int, float)):
                            predictions.append(
                                {
                                    "value": pred,
                                    "step": f"final_{i}",
                                    "question": "",
                                    "response": "",
                                    "raw_response": "",
                                }
                            )

            # Check if any predictions are extreme
            for pred in predictions:
                if abs(pred["value"]) > threshold:
                    extreme_cases.append(
                        {
                            "file": str(json_file.relative_to(results_dir)),
                            "prediction": pred["value"],
                            "step": pred["step"],
                            "question": pred["question"][:100] if pred["question"] else "",
                            "response": pred["response"][:200] if pred["response"] else "",
                            "raw_response": pred["raw_response"][:500]
                            if pred["raw_response"]
                            else "",
                            "metadata": {
                                "env": data.get("config", {}).get("envs", "?"),
                                "model": data.get("config", {}).get("llms", "?"),
                                "seed": data.get("config", {}).get("seed", "?"),
                            },
                        }
                    )

        except Exception:
            # Skip files that can't be read
            pass

    return extreme_cases


def analyze_specific_value(
    results_dir: Path, target_value: float = 67219139
) -> list[dict[str, Any]]:
    """Search for a specific prediction value."""
    matches = []

    json_files = list(results_dir.rglob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file) as f:
                content = f.read()

            # Simple string search for the value
            if str(int(target_value)) in content:
                print(f"  ✅ Found {target_value:,.0f} in {json_file.name}")

                # Parse and extract context
                data = json.loads(content)

                # Extract the relevant step
                if "step_results" in data:
                    for step in data["step_results"]:
                        if "prediction" in step and abs(step["prediction"] - target_value) < 1:
                            matches.append(
                                {
                                    "file": str(json_file.relative_to(results_dir)),
                                    "step": step.get("step", "?"),
                                    "prediction": step["prediction"],
                                    "question": step.get("question", ""),
                                    "response": step.get("response", ""),
                                    "raw_response": step.get("raw_response", ""),
                                    "metadata": data.get("config", {}),
                                }
                            )

        except Exception:
            pass

    return matches


def extract_response_context(match: dict[str, Any]) -> None:
    """Print detailed context for a match."""
    print("\n" + "=" * 80)
    print(f"📄 File: {match['file']}")
    print(f"🎯 Prediction: {match['prediction']:,.0f}")
    print(f"📊 Step: {match['step']}")
    print(f"🔧 Model: {match['metadata'].get('llms', '?')}")
    print(f"🌍 Environment: {match['metadata'].get('envs', '?')}")
    print(f"🎲 Seed: {match['metadata'].get('seed', '?')}")
    print("=" * 80)

    if match.get("question"):
        print(f"\n❓ Question:\n{match['question']}\n")

    if match.get("raw_response"):
        print(f"💬 Raw LLM Response:\n{match['raw_response']}\n")
    elif match.get("response"):
        print(f"💬 Response:\n{match['response']}\n")


def main():
    """Main analysis."""
    results_dir = Path(__file__).parent.parent / "results"

    print("=" * 80)
    print("🔍 ANALYZING LOCAL RESULTS FOR EXTREME PREDICTIONS")
    print("=" * 80)

    # Strategy 1: Find all extreme predictions
    print("\n📊 Searching for extreme predictions (> 1,000,000)...")
    extreme = find_extreme_predictions(results_dir, threshold=1e6)

    if extreme:
        print(f"\n✅ Found {len(extreme)} extreme predictions")

        # Show summary
        df = pd.DataFrame(
            [
                {
                    "file": e["file"].split("/")[-1][:50],
                    "prediction": f"{e['prediction']:,.0f}",
                    "model": e["metadata"]["model"],
                    "env": e["metadata"]["env"],
                }
                for e in extreme
            ]
        )
        print("\n" + df.to_string(index=False))

        # Show details for top cases
        print("\n" + "=" * 80)
        print("📝 DETAILED CONTEXT FOR EXTREME CASES")
        print("=" * 80)

        for match in extreme[:5]:  # Top 5
            extract_response_context(match)

    else:
        print("❌ No extreme predictions found")

    # Strategy 2: Search for specific value
    print("\n" + "=" * 80)
    print("🎯 Searching for specific value: 67,219,139")
    print("=" * 80)

    specific_matches = analyze_specific_value(results_dir, target_value=67219139)

    if specific_matches:
        print(f"\n✅ Found {len(specific_matches)} matches for 67,219,139")

        for match in specific_matches:
            extract_response_context(match)

    else:
        print("❌ No matches found for 67,219,139")

    # Strategy 3: Analyze peregrines specifically
    print("\n" + "=" * 80)
    print("🦅 Analyzing peregrines results specifically")
    print("=" * 80)

    peregrines_dir = results_dir / "peregrines"
    if peregrines_dir.exists():
        peregrine_extreme = find_extreme_predictions(peregrines_dir, threshold=1e5)
        print(f"  Found {len(peregrine_extreme)} extreme predictions in peregrines")

        for match in peregrine_extreme[:3]:
            print(f"\n  File: {match['file']}")
            print(f"  Prediction: {match['prediction']:,.0f}")
            print(f"  Model: {match['metadata']['model']}")

    # Save detailed results
    output_path = results_dir.parent / "data" / "local_extreme_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "extreme_predictions": extreme,
                "specific_value_matches": specific_matches,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
