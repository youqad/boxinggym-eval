import json
import os
import glob
from openai import OpenAI
import csv
import re
from typing import Dict, List
from collections import defaultdict

# Configuration
RESULTS_DIR = "/sailhome/agam/scr_agam/boxing-gym/results"

# Only evaluate these 5 environments
TARGET_ENVIRONMENTS = ["death_process", "dugongs", "emotion", "hyperbolic_temporal_discount", "morals"]

# Initialize OpenAI client
client = OpenAI(api_key="")

def find_experiment_files(results_dir: str) -> List[str]:
    """Find experiment files for gpt-4o and claude agents only (no boxloop)."""
    pattern = os.path.join(results_dir, "*/direct_discovery_*_discovery_*_*.json")
    all_files = glob.glob(pattern)
    
    filtered_files = []
    for file in all_files:
        filename = os.path.basename(file)
        env_name = file.split('/')[-2]
        
        # Filter criteria
        if (env_name in TARGET_ENVIRONMENTS and 
            ("qwen2.5-7b" in filename or "OpenThinker-7B" in filename) and 
            "boxloop" not in filename):
            filtered_files.append(file)
    
    return filtered_files

def parse_filename(filepath: str) -> Dict[str, str]:
    """Extract environment, model, and prior info from filename."""
    path_parts = filepath.split('/')
    env_name = path_parts[-2]
    filename = path_parts[-1].replace('.json', '')
    
    # Parse filename: direct_discovery_{model}_discovery_{prior}_{number}
    parts = filename.split('_')
    
    # Find model (everything between direct_discovery and discovery)
    model_start = 2  # after "direct_discovery"
    model_parts = []
    for i in range(model_start, len(parts)):
        if parts[i] == "discovery":
            break
        model_parts.append(parts[i])
    
    model = "_".join(model_parts)
    
    # Find prior (True/False after discovery)
    prior = "unknown"
    for i, part in enumerate(parts):
        if part == "discovery" and i + 1 < len(parts):
            prior = parts[i + 1]
            break
    
    return {
        'env_name': env_name,
        'model': model,
        'prior': prior,
        'filepath': filepath
    }

def extract_scientist_explanation(data: Dict) -> str:
    """Extract scientist's final explanation from JSON data."""
    # Look in scientist_messages for the final explanation
    scientist_messages = data.get('scientist_messages', [])
    
    for msg in reversed(scientist_messages):
        if 'role:assistant' in msg:
            content = msg.split('messaage:')[-1].strip()
            # Look for the explanation request response (usually long and detailed)
            if len(content) > 150 and ('environment' in content.lower() or 'function' in content.lower()):
                return content
    
    # Fallback to data.explanations
    if 'data' in data and 'explanations' in data['data']:
        explanations = data['data']['explanations']
        if explanations:
            return explanations[0]
    
    return ""

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())

def evaluate_faithfulness(scientist_explanation: str, naive_messages: List[str]) -> Dict:
    """Evaluate faithfulness using GPT-4."""
    naive_responses = []
    for msg in naive_messages:
        if 'role:assistant' in msg:
            response = msg.split('messaage:')[-1].strip()
            naive_responses.append(response)
    
    naive_text = "\n\n".join(naive_responses)
    
    prompt = f""" Briefly evaluate how well the naive agent followed the scientist's explanation.

SCIENTIST'S EXPLANATION:
{scientist_explanation}

NAIVE AGENT'S RESPONSES:
{naive_text}

Rate FAITHFULNESS (1-5): How well did the naive agent follow the scientist's explanation?
- 5: Perfect adherence, uses specific patterns/rules mentioned
- 4: Mostly follows with minor deviations
- 3: Generally follows but with some inconsistencies
- 2: Partially follows with significant deviations
- 1: Rarely follows, makes up own rules

<faithfulness>X</faithfulness>

Brief reasoning: [Why you gave this score]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        faithfulness_match = re.search(r'<faithfulness>(\d+)</faithfulness>', content)
        
        if faithfulness_match:
            score = int(faithfulness_match.group(1))
            return {
                'faithfulness': score,
                'reasoning': content,
                'success': True
            }
        else:
            # Score not found in response
            print(f"Could not extract faithfulness score from response: {content[:200]}...")
            return {
                'reasoning': content,
                'success': False,
                'error': 'Score not found in response'
            }
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def evaluate_conciseness(scientist_explanation: str) -> Dict:
    """Evaluate conciseness of scientist explanation."""
    prompt = f"""Briefly evaluate how concise and efficient this scientific explanation is.

EXPLANATION:
{scientist_explanation}

EXAMPLES FOR REFERENCE:

Example 1 - HIGH CONCISENESS (5):
"The function increases linearly from 0 to 1 (0.5→10, 1.0→20), then decreases linearly from 1 to 2 (1.5→15, 2.0→5)."
✓ Gets straight to the point, no unnecessary words, clear patterns

Example 2 - MEDIUM CONCISENESS (3):
"Based on observations, the function appears to have linear behavior in two ranges. In the first range from 0 to 1, the output increases as input increases. For example, input 0.5 gives 10 and input 1.0 gives 20. In the second range from 1 to 2, the output decreases as input increases."
~ Some repetition and wordiness, but covers key points

Example 3 - LOW CONCISENESS (1):
"The environment you are dealing with is quite interesting and complex. After conducting various experiments and observations across different input ranges, I have discovered that there are several distinct behavioral patterns that emerge. The function exhibits different characteristics in different ranges, and I will now explain each of these ranges in detail..."
✗ Verbose, repetitive, takes too long to get to the point

Rate CONCISENESS (1-5): How efficiently does this explanation convey the key information?
- 5: Extremely concise, every word adds value, no redundancy
- 4: Mostly concise with minimal unnecessary content
- 3: Generally efficient but some wordiness or repetition
- 2: Somewhat wordy with noticeable redundancy
- 1: Very verbose, lots of unnecessary content, hard to extract key points

<conciseness>X</conciseness>

Brief reasoning: [Why you gave this score, noting specific examples of efficiency or wordiness]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        content = response.choices[0].message.content
        conciseness_match = re.search(r'<conciseness>(\d+)</conciseness>', content)
        
        if conciseness_match:
            score = int(conciseness_match.group(1))
            return {
                'conciseness': score,
                'reasoning': content,
                'success': True
            }
        else:
            print(f"Could not extract conciseness score from response: {content[:200]}...")
            return {
                'reasoning': content,
                'success': False,
                'error': 'Score not found in response'
            }
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def evaluate_readability(scientist_explanation: str) -> Dict:
    """Evaluate readability of scientist explanation."""
    prompt = f"""Briefly evaluate the clarity and readability of this scientific explanation.

EXPLANATION:
{scientist_explanation}

EXAMPLES FOR REFERENCE:

Example 1 - HIGH READABILITY (5):
"The function has three distinct ranges:
1. Range 0-1: Output increases (0.5→8, 0.75→13, 1.0→19)
2. Range 1-1.5: Output plateaus around 19, with peak at 1.25→22  
3. Range 1.5-2: Output increases again (1.55→25, 1.6→22)"
✓ Well-structured, clear examples, easy to follow logic

Example 2 - MEDIUM READABILITY (3):
"Based on observations, the function appears to have non-linear behavior. The output increases in some ranges but not others. There are peaks and plateaus that make prediction challenging."
~ Generally clear but lacks specific guidance and structure

Example 3 - LOW READABILITY (1):
"It's complicated with various behaviors happening and the patterns aren't straightforward to determine what will happen next since different inputs give different outputs in ways that are hard to predict."
✗ Vague, no structure, confusing, little actionable information

Rate READABILITY (1-5): How clear and understandable is this explanation for someone making predictions?
- 5: Extremely clear, well-structured, specific guidance with examples
- 4: Clear with good structure and helpful details
- 3: Generally clear but could be more specific or better organized
- 2: Somewhat unclear, vague, or poorly structured  
- 1: Very unclear, confusing, little actionable guidance

<readability>X</readability>

Brief reasoning: [Why you gave this score, noting structure, clarity, and usefulness for predictions]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        content = response.choices[0].message.content
        readability_match = re.search(r'<readability>(\d+)</readability>', content)
        
        if readability_match:
            score = int(readability_match.group(1))
            return {
                'readability': score,
                'reasoning': content,
                'success': True
            }
        else:
            print(f"Could not extract readability score from response: {content[:200]}...")
            return {
                'reasoning': content,
                'success': False,
                'error': 'Score not found in response'
            }
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    """Main execution function."""
    files = find_experiment_files(RESULTS_DIR)
    print(f"Found {len(files)} experiment files")
    
    results = []
    
    for filepath in files:
        print(f"Processing: {os.path.basename(filepath)}")
        
        # Parse file info
        file_info = parse_filename(filepath)
        
        # Load data
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
        
        # Extract scientist explanation
        scientist_explanation = extract_scientist_explanation(data)
        if not scientist_explanation:
            print(f"No explanation found in {filepath}")
            continue
        
        word_count = count_words(scientist_explanation)
        naive_messages = data.get('naive_messages', [])
        
        result = {
            'env_name': file_info['env_name'],
            'model': file_info['model'],
            'prior': file_info['prior'],
            'word_count': word_count,
            'filepath': filepath,
            'scientist_explanation': scientist_explanation  # Save explanation for debugging
        }
        
        # Evaluate faithfulness for all 5 environments
        print(f"Evaluating faithfulness for {file_info['env_name']}...")
        faithfulness_eval = evaluate_faithfulness(scientist_explanation, naive_messages)
        if faithfulness_eval['success']:
            result['faithfulness'] = faithfulness_eval['faithfulness']
            result['faithfulness_reasoning'] = faithfulness_eval['reasoning']
        else:
            result['faithfulness_error'] = faithfulness_eval['error']
            print(f"Faithfulness evaluation failed: {faithfulness_eval['error']}")
        
        # Evaluate conciseness and readability only for death_process
        if file_info['env_name'] == 'death_process':
            print(f"Evaluating conciseness for death_process...")
            conciseness_eval = evaluate_conciseness(scientist_explanation)
            if conciseness_eval['success']:
                result['conciseness'] = conciseness_eval['conciseness']
                result['conciseness_reasoning'] = conciseness_eval['reasoning']
            else:
                result['conciseness_error'] = conciseness_eval['error']
                print(f"Conciseness evaluation failed: {conciseness_eval['error']}")
            
            print(f"Evaluating readability for death_process...")
            readability_eval = evaluate_readability(scientist_explanation)
            if readability_eval['success']:
                result['readability'] = readability_eval['readability']
                result['readability_reasoning'] = readability_eval['reasoning']
            else:
                result['readability_error'] = readability_eval['error']
                print(f"Readability evaluation failed: {readability_eval['error']}")
        
        results.append(result)
    
    # Calculate averages
    averages = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for result in results:
        env = result['env_name']
        model = result['model']
        prior = result['prior']
        
        if 'faithfulness' in result and result['faithfulness'] is not None:
            averages[env][model][prior].append(result['faithfulness'])
        
        # For death_process, also collect conciseness, readability and word count
        if env == 'death_process':
            if 'conciseness' in result and result['conciseness'] is not None:
                averages[f"{env}_conciseness"][model][prior].append(result['conciseness'])
            if 'readability' in result and result['readability'] is not None:
                averages[f"{env}_readability"][model][prior].append(result['readability'])
            if result['word_count'] is not None:  # Word count should always be valid, but just in case
                averages[f"{env}_wordcount"][model][prior].append(result['word_count'])
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    successful_faithfulness = len([r for r in results if 'faithfulness' in r])
    successful_conciseness = len([r for r in results if 'conciseness' in r])
    successful_readability = len([r for r in results if 'readability' in r])
    print(f"Total files processed: {len(results)}")
    print(f"Successful faithfulness evaluations: {successful_faithfulness}")
    print(f"Successful conciseness evaluations: {successful_conciseness}")
    print(f"Successful readability evaluations: {successful_readability}")
    
    # Print any failures
    failures = [r for r in results if 'faithfulness_error' in r or 'conciseness_error' in r or 'readability_error' in r]
    if failures:
        print(f"\nFailed evaluations: {len(failures)}")
        for failure in failures:
            errors = []
            if 'faithfulness_error' in failure:
                errors.append(f"faithfulness: {failure['faithfulness_error']}")
            if 'conciseness_error' in failure:
                errors.append(f"conciseness: {failure['conciseness_error']}")
            if 'readability_error' in failure:
                errors.append(f"readability: {failure['readability_error']}")
            print(f"  {failure['env_name']}/{failure['model']}: {'; '.join(errors)}")
    
    # Print results
    print("\n" + "="*60)
    print("FAITHFULNESS SCORES (All 5 Environments)")
    print("="*60)
    
    for env in TARGET_ENVIRONMENTS:
        if env in averages:
            print(f"\n{env.upper()}:")
            for model in averages[env]:
                for prior in averages[env][model]:
                    scores = [s for s in averages[env][model][prior] if s is not None]  # Filter out None
                    if scores:  # Only calculate average if we have valid scores
                        avg = sum(scores) / len(scores)
                        print(f"  {model} (prior={prior}): {avg:.2f} (n={len(scores)})")
                    else:
                        print(f"  {model} (prior={prior}): No valid scores")
    
    print("\n" + "="*60)
    print("DEATH PROCESS: WORD COUNT, CONCISENESS & READABILITY")
    print("="*60)
    
    # Word count (raw measure)
    wordcount_key = 'death_process_wordcount'
    if wordcount_key in averages:
        print(f"\nWORD COUNT (Raw Measure):")
        for model in averages[wordcount_key]:
            for prior in averages[wordcount_key][model]:
                counts = [c for c in averages[wordcount_key][model][prior] if c is not None]
                if counts:
                    avg = sum(counts) / len(counts)
                    print(f"  {model} (prior={prior}): {avg:.1f} words (n={len(counts)})")
                else:
                    print(f"  {model} (prior={prior}): No valid word counts")
    
    # Conciseness (1-5 scale)
    conciseness_key = 'death_process_conciseness'
    if conciseness_key in averages:
        print(f"\nCONCISENESS (1-5 Scale):")
        for model in averages[conciseness_key]:
            for prior in averages[conciseness_key][model]:
                scores = [s for s in averages[conciseness_key][model][prior] if s is not None]
                if scores:
                    avg = sum(scores) / len(scores)
                    print(f"  {model} (prior={prior}): {avg:.2f} (n={len(scores)})")
                else:
                    print(f"  {model} (prior={prior}): No valid scores")
    
    # Readability (1-5 scale)
    readability_key = 'death_process_readability'
    if readability_key in averages:
        print(f"\nREADABILITY (1-5 Scale):")
        for model in averages[readability_key]:
            for prior in averages[readability_key][model]:
                scores = [s for s in averages[readability_key][model][prior] if s is not None]
                if scores:
                    avg = sum(scores) / len(scores)
                    print(f"  {model} (prior={prior}): {avg:.2f} (n={len(scores)})")
                else:
                    print(f"  {model} (prior={prior}): No valid scores")
    
    # Save detailed results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to evaluation_results.json")

if __name__ == "__main__":
    main()