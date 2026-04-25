# Google Code Golf Community Best 

- **Author:** Tony Li
- **Votes:** 113
- **Ref:** tonylica/google-code-golf-community-best
- **URL:** https://www.kaggle.com/code/tonylica/google-code-golf-community-best
- **Last run:** 2025-10-30 16:57:15.480000

---

```python
import warnings
warnings.filterwarnings("ignore")

!pip install zopfli
```

```python
# Auto-detect environment: local vs Kaggle
import os

if os.path.exists("./tasks_others"):
    # Local environment
    print("Detected LOCAL environment - using local tasks folder")
    ENVIRONMENT = "local"
    sources = [
        "./tasks_others/tasks",
        "./tasks_others/tasks_public",
    ]
else:
    # Kaggle environment - treat input paths as direct dataset sources
    print("Detected KAGGLE environment - using dataset sources")
    ENVIRONMENT = "kaggle"
    sources = [
        "/kaggle/input/google-code-golf-2025-submit",
        "/kaggle/input/dead-code"
    ]

print(f"Environment: {ENVIRONMENT}")
print(f"Sources: {sources}")
```

Remove failing solutions

```python
failing = []

# Tasks that take too long to verify - skip verification for these
slow_verification_tasks = [157]
```

The zipper:

```python
from zipfile import ZipFile
import zipfile
import zopfli.zlib
import zlib

def zip_src(src_code):
    candidates=[src_code]
    for compress in[zopfli.zlib.compress,lambda d:zlib.compress(d,9)]:
        for trailing in[b'',b'\n']:
            src=src_code+trailing
            while(comp:=compress(src))[-1]==ord('"'):src+=b'#'
            for delim in[b"'",b'"']:
                esc_map={0:b'\\x00',ord('\n'):b'\\n',ord('\r'):b'\\r',ord('\\'):b'\\\\',delim[0]:b'\\'+delim}
                sanitized=b''.join(esc_map.get(b,bytes([b]))for b in comp)
                compressed=b'import zlib\nexec(zlib.decompress(bytes('+delim+sanitized+delim+b',"L1")))'
                if max(sanitized)>127:compressed=b'#coding:L1\n'+compressed
                else:print('no header needed!')
                candidates.append(compressed)
            esc_map={0:b'\\x00',ord('\r'):b'\\r',ord('\\'):b'\\\\'}
            sanitized=b''.join(esc_map.get(b,bytes([b]))for b in comp)
            compressed=b'import zlib\nexec(zlib.decompress(bytes("""'+sanitized+b'""","L1")))'
            if max(sanitized)>127:compressed=b'#coding:L1\n'+compressed
            else:print('no header needed!')
            candidates.append(compressed)
    valid_options=[]
    for code in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SyntaxWarning)
                with open('tmp.py','wb')as f:f.write(code)
                with open('tmp.py','rb')as f:x=f.read()
                exec(x,{})
                valid_options.append(code)
        except:0
    return min(valid_options,key=len)
```

And get zipping

```python
! pip install python-minifier

import os
from python_minifier import minify
import warnings
import traceback
import zipfile
import shutil
from collections import defaultdict
import json
import copy
import numpy as np
import re

warnings.filterwarnings("ignore")

def is_already_compressed(source_code):
    """Check if source code is already zlib compressed"""
    return "exec(zlib.decompress" in source_code

def load_task_examples(task_num):
    """Load examples for a task (simplified version of verify1.py logic)"""
    # Try to find task data file
    task_data_paths = [
        f"/kaggle/input/google-code-golf-2025/task{task_num:03d}.json",
        f"../google-code-golf-2025/task{task_num:03d}.json"
    ]
    
    for path in task_data_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    
    print(f"Warning: No test data found for task {task_num}")
    return None

def verify_solution_code(solution_code, task_num, examples=None):
    """Verify if a solution works correctly"""
    if examples is None:
        examples = load_task_examples(task_num)
        if examples is None:
            return False, "No test data available"
    
    try:
        # Execute the solution code
        namespace = {}
        exec(solution_code, namespace)
        
        if 'p' not in namespace:
            return False, "No function 'p' found"
        
        solution_func = namespace['p']
        if not callable(solution_func):
            return False, "Function 'p' is not callable"
        
        # Test on all available test cases
        test_cases = []
        for category in ['test', 'train', 'arc-gen']:
            if category in examples:
                test_cases.extend(examples[category])
        
        if not test_cases:
            return False, "No test cases available"
        
        correct_count = 0
        total_count = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            try:
                result = solution_func(copy.deepcopy(test_case['input']))
                
                # Convert result to comparable format
                result_json = json.dumps(result)
                result_json = result_json.replace("true", "1").replace("false", "0")
                
                # Check for unsafe characters
                unsafe_chars = re.compile(r"[^0-9,\[\]\s\.]")
                if unsafe_chars.search(result_json):
                    return False, f"Invalid output format: {result_json[:100]}"
                
                result_array = np.array(json.loads(result_json))
                expected_array = np.array(test_case['output'])
                
                if np.array_equal(result_array, expected_array):
                    correct_count += 1
                else:
                    return False, f"Wrong output on test case {i+1}/{total_count}"
                    
            except Exception as e:
                return False, f"Exception on test case {i+1}: {str(e)[:100]}"
        
        return True, f"Passed all {correct_count}/{total_count} test cases"
        
    except Exception as e:
        return False, f"Execution error: {str(e)[:100]}"

def safe_minify(source_code, file_path="unknown"):
    """Error-proof minification that reports bytes saved (latin-1 only)"""
    original_size = len(source_code.encode('latin-1'))
    try:
        minified = minify(source_code)
        minified_size = len(minified.encode('latin-1'))
        bytes_saved = original_size - minified_size
        
        # If minified result is bigger than original, use original instead
        if minified_size > original_size:
            return source_code, 0, None
        
        return minified, bytes_saved, None
    except Exception as e:
        # Return original code if minification fails
        error_msg = f"File: {file_path} | Error: {str(e)}"
        return source_code, 0, error_msg

print(f"\nUsing sources: {sources}")

total_save = 0
minify_save = 0
minify_errors = 0
minify_error_details = []
already_compressed_count = 0  # Track how many scripts were already compressed
score = 0

# Track verification statistics
total_verification_attempted = 0
total_verification_failed = 0
tasks_with_verification_issues = 0
default_verification_attempted = 0
default_verification_failed = 0
failed_default_tasks = []  # Track which default source tasks failed verification

# Track source usage and savings
source_stats = defaultdict(lambda: {'tasks_used': 0, 'bytes_saved': 0, 'original_bytes': 0, 'final_bytes': 0, 'already_compressed_tasks': 0})
task_source_map = {}  # Track which source was used for each task
non_default_tasks = {}  # Track tasks that used non-default sources
default_source = sources[0] if sources else None  # First source is default

if ENVIRONMENT == "kaggle":
    submission = "/kaggle/working/submission"
else:
    submission = "./submission"

os.makedirs(submission, exist_ok=True)

print(f"\nProcessing 400 tasks...")

for task_num in range(1, 401):
    path_out = f"{submission}/task{task_num:03d}.py" 

    # Collect all candidates with their optimized versions
    candidates = []
    
    # Check all sources for candidate solutions
    for source in sources:
        path_in = f"{source}/task{task_num:03d}.py"
        if not os.path.exists(path_in): continue
        if path_in in failing: continue

        # Read all files as latin-1 (binary hack)
        try:
            with open(path_in, "r", encoding='latin-1') as task_in:
                new_src = task_in.read()
        except Exception as e:
            print(f"Error reading {path_in}: {e}")
            continue
        
        # Remove BOM if present to save bytes
        if new_src.startswith('\ufeff'):
            new_src = new_src[1:]
        elif new_src.startswith('\xff\xfe') or new_src.startswith('\xfe\xff'):
            new_src = new_src[2:]
        
        original_size = len(new_src.encode('latin-1'))
        
        # Check if the script is already zlib compressed
        if is_already_compressed(new_src):
            already_compressed_count += 1
            final_size = original_size  # Use actual file size, no optimization needed
            bytes_saved = 0  # No minification savings since we skipped it
            compression_improvement = 0
        else:
            # Try safe minification (latin-1 only)
            minified_src, bytes_saved, error = safe_minify(new_src, path_in)
            if error:
                minify_errors += 1
                minify_error_details.append(error)
                print(f"Minification failed for task {task_num}: {error}")
            else:
                minify_save += bytes_saved
            
            # Get the best of raw vs minified (latin-1 encoding)
            raw_size = len(new_src.encode('latin-1'))
            minified_size = len(minified_src.encode('latin-1'))
            
            # Use the smaller version for compression
            candidate_src = minified_src.encode('latin-1') if minified_size < raw_size else new_src.encode('latin-1')
            
            # Apply compression to get final size
            compressed_src = zip_src(candidate_src)
            final_size = len(compressed_src) if len(compressed_src) < len(candidate_src) else len(candidate_src)
            
            # Track compression savings for this candidate
            compression_improvement = len(candidate_src) - len(compressed_src) if len(compressed_src) < len(candidate_src) else 0
        
        # Store candidate info
        candidates.append({
            'source': source,
            'original_src': new_src,  # Keep for verification only
            'original_size': original_size,
            'final_size': final_size,  # Estimated optimized size
            'compression_improvement': compression_improvement,
            'was_already_compressed': is_already_compressed(new_src),
            'path': path_in  # Keep original file path for binary copying
        })
    
    # Sort candidates by final size (smallest first)
    candidates.sort(key=lambda x: x['final_size'])
    
    # Find the default candidate (first source is default)
    default_candidate = None
    for candidate in candidates:
        if candidate['source'] == default_source:
            default_candidate = candidate
            break
    
    if not default_candidate:
        print(f"Task {task_num}: No default solution found, using smallest available")
        # Fallback to smallest if no default found
        default_candidate = candidates[0] if candidates else None
    
    if not default_candidate:
        print(f"Task {task_num}: No candidates found at all")
        continue
    
    # Find smaller candidates than default
    smaller_candidates = [c for c in candidates if c['final_size'] < default_candidate['final_size']]
    
    # First, verify the default solution works
    default_verification_attempted += 1
    total_verification_attempted += 1
    
    # Load test examples once for this task
    examples = load_task_examples(task_num)
    
    # Verify default solution
    if task_num in slow_verification_tasks:
        print(f"Task {task_num}: ⏭ Skipping default verification (known to be slow) - assuming it works")
        default_is_working = True
        default_message = "Verification skipped (slow task)"
    else:
        print(f"Task {task_num}: Verifying default solution from {default_candidate['source']}")
        default_is_working, default_message = verify_solution_code(default_candidate['original_src'], task_num, examples)
    
    if not default_is_working:
        # Default solution failed verification
        default_verification_failed += 1
        total_verification_failed += 1
        failed_default_tasks.append({
            'task_num': task_num,
            'source': default_candidate['source'],
            'error': default_message
        })
        print(f"  ✗ Default solution failed: {default_message}")
        
        # Remove default from candidates and find next best working solution
        working_candidates = [c for c in candidates if c != default_candidate]
        working_candidates.sort(key=lambda x: x['final_size'])  # Sort by size
        
        best_candidate = None
        for i, candidate in enumerate(working_candidates):
            total_verification_attempted += 1
            print(f"  Testing fallback candidate {i+1}/{len(working_candidates)}: {candidate['source']} ({candidate['final_size']} bytes)")
            
            if task_num in slow_verification_tasks:
                print(f"    ⏭ Skipping verification (known to be slow) - assuming it works")
                is_working = True
                message = "Verification skipped (slow task)"
            else:
                is_working, message = verify_solution_code(candidate['original_src'], task_num, examples)
            
            if is_working:
                best_candidate = candidate
                print(f"    ✓ Working! Using fallback {candidate['source']} solution ({candidate['final_size']} bytes)")
                break
            else:
                total_verification_failed += 1
                print(f"    ✗ Failed: {message}")
        
        if not best_candidate:
            print(f"  ⚠ No working solution found for task {task_num}! Using default anyway (may fail)")
            best_candidate = default_candidate
    else:
        # Default solution works
        print(f"  ✓ Default solution verified successfully")
        best_candidate = default_candidate
    
    # Initialize best solution variables
    best_source = best_candidate['source']
    best_original_size = best_candidate['original_size']
    best_compression_savings = best_candidate['compression_improvement']
    best_was_already_compressed = best_candidate['was_already_compressed']
    verification_attempted = 0
    verification_failed = 0
    
    # Only test smaller candidates if default solution worked and there are smaller alternatives
    if default_is_working and smaller_candidates:
        print(f"Task {task_num}: Default size {default_candidate['final_size']}, testing {len(smaller_candidates)} smaller alternatives")
        
        # Test smaller candidates in order (smallest first) until we find a working one
        for i, candidate in enumerate(smaller_candidates):
            verification_attempted += 1
            total_verification_attempted += 1
            
            print(f"  Testing candidate {i+1}/{len(smaller_candidates)}: {candidate['source']} ({candidate['final_size']} bytes)")
            
            # Skip verification for slow tasks
            if task_num in slow_verification_tasks:
                print(f"  ⏭ Skipping verification for task {task_num} (known to be slow) - assuming it works")
                is_working = True
                message = "Verification skipped (slow task)"
            else:
                # Verify the solution works
                is_working, message = verify_solution_code(candidate['original_src'], task_num, examples)
            
            if is_working:
                # This smaller candidate works, use it instead of default
                best_candidate = candidate
                best_source = candidate['source']
                best_original_size = candidate['original_size']
                best_compression_savings = candidate['compression_improvement']
                best_was_already_compressed = candidate['was_already_compressed']
                
                bytes_saved = default_candidate['final_size'] - candidate['final_size']
                print(f"  ✓ Working! Using {candidate['source']} solution ({candidate['final_size']} bytes, saved {bytes_saved} bytes)")
                break
            else:
                verification_failed += 1
                total_verification_failed += 1
                print(f"  ✗ Failed: {message}")
        
        # Track tasks that had verification issues
        if verification_failed > 0:
            tasks_with_verification_issues += 1
        
        # If no smaller working solution found, we already have default set above
        if verification_attempted > 0 and best_candidate == default_candidate:
            print(f"  Using default solution ({default_candidate['final_size']} bytes) - no smaller working solution found")
    
    # Add compression savings for the chosen source
    if best_compression_savings > 0:
        total_save += best_compression_savings
    
    # Track which source was used for this task
    if best_source:
        task_source_map[task_num] = best_source
        source_stats[best_source]['tasks_used'] += 1
        source_stats[best_source]['original_bytes'] += best_original_size
        
        # Track final size and total bytes saved for the source that was used
        source_stats[best_source]['final_bytes'] += best_candidate['final_size']
        source_stats[best_source]['bytes_saved'] += best_original_size - best_candidate['final_size']
        
        # Track if this was an already-compressed script
        if best_was_already_compressed:
            source_stats[best_source]['already_compressed_tasks'] += 1
        
        # Track if non-default source was used
        if best_source != default_source:
            # Process default source through all optimization stages for comparison
            default_file_path = f"{default_source}/task{task_num:03d}.py"
            task_file_path = f"{best_source}/task{task_num:03d}.py"
            
            if os.path.exists(task_file_path):
                try:
                    # Read as latin-1 (binary hack)
                    with open(task_file_path, "r", encoding='latin-1') as f:
                        content = f.read()
                    
                    # Remove BOM if present to save bytes
                    if content.startswith('\ufeff'):
                        content = content[1:]
                    elif content.startswith('\xff\xfe') or content.startswith('\xfe\xff'):
                        content = content[2:]
                        
                except Exception as e:
                    print(f"Error reading chosen source for task {task_num}: {e}")
                    continue
                
                # Process default source through all stages if it exists
                default_stages = {
                    'raw_size': 0,
                    'minified_size': 0,
                    'compressed_size': 0,
                    'minify_error': None
                }
                
                if os.path.exists(default_file_path):
                    try:
                        # Read as latin-1 (binary hack)
                        with open(default_file_path, "r", encoding='latin-1') as f:
                            default_content = f.read()
                        
                        # Remove BOM if present to save bytes
                        if default_content.startswith('\ufeff'):
                            default_content = default_content[1:]
                        elif default_content.startswith('\xff\xfe') or default_content.startswith('\xfe\xff'):
                            default_content = default_content[2:]
                            
                    except Exception as e:
                        print(f"Error reading default source for task {task_num}: {e}")
                        continue
                    
                    default_stages['raw_size'] = len(default_content.encode('latin-1'))
                    
                    # Check if default source is already compressed
                    if is_already_compressed(default_content):
                        # Already compressed - skip optimization stages
                        default_stages['minified_size'] = default_stages['raw_size']  # No minification applied
                        default_stages['compressed_size'] = default_stages['raw_size']  # Already compressed
                        default_minified = default_content  # Keep original
                    else:
                        # Try minifying default source
                        default_minified, _, default_error = safe_minify(default_content, default_file_path)
                        if default_error:
                            default_stages['minify_error'] = default_error
                            default_stages['minified_size'] = default_stages['raw_size']  # Use raw size if minify fails
                        else:
                            default_stages['minified_size'] = len(default_minified.encode('latin-1'))
                        
                        # Try compressing default source (use best of minified vs raw)
                        default_best_src = min([default_minified.encode('latin-1'), default_content.encode('latin-1')], key=len)
                        default_compressed = zip_src(default_best_src)
                        default_compressed_improvement = len(default_best_src) - len(default_compressed)
                        
                        if default_compressed_improvement > 0:
                            default_stages['compressed_size'] = len(default_compressed)
                        else:
                            default_stages['compressed_size'] = len(default_best_src)
                
                # Current source stages - calculate all optimization steps properly
                current_stages = {
                    'raw_size': len(content.encode('latin-1')),
                    'minified_size': 0,
                    'compressed_size': 0,
                    'minify_error': None
                }
                
                # Check if current source is already compressed
                if is_already_compressed(content):
                    # Already compressed - skip optimization stages
                    current_stages['minified_size'] = current_stages['raw_size']  # No minification applied
                    current_stages['compressed_size'] = current_stages['raw_size']  # Already compressed
                    current_minified = content  # Keep original
                else:
                    # We already processed this through minification above, but let's get the exact sizes
                    current_minified, _, current_error = safe_minify(content, task_file_path)
                    if current_error:
                        current_stages['minify_error'] = current_error
                        current_stages['minified_size'] = current_stages['raw_size']
                    else:
                        current_stages['minified_size'] = len(current_minified.encode('latin-1'))
                    
                    # Calculate compressed size for current source
                    current_best_src = min([current_minified.encode('latin-1'), content.encode('latin-1')], key=len)
                    current_compressed = zip_src(current_best_src)
                    current_compressed_improvement = len(current_best_src) - len(current_compressed)
                    
                    if current_compressed_improvement > 0:
                        current_stages['compressed_size'] = len(current_compressed)
                    else:
                        current_stages['compressed_size'] = len(current_best_src)
                
                non_default_tasks[task_num] = {
                    'source': best_source,
                    'content': content,
                    'file_path': task_file_path,
                    'default_stages': default_stages,
                    'current_stages': current_stages,
                    'final_bytes_saved': default_stages['compressed_size'] - current_stages['compressed_size']
                }
    
    # Compression was already applied during source selection
    # Source statistics tracking is now handled in the source selection loop above
    
    score += max(1, 2500-best_candidate['final_size'])
    
    # Write the final optimized content (or original if compressed)
    if best_candidate['was_already_compressed']:
        # Copy compressed file as-is (binary)
        with open(best_candidate['path'], "rb") as src_file:
            with open(path_out, "wb") as task_out:
                task_out.write(src_file.read())
    else:
        # Write optimized content for non-compressed files
        # This requires re-doing the optimization to get the actual optimized bytes
        src_content = best_candidate['original_src']
        
        # Re-do minification
        minified_src, _, error = safe_minify(src_content, best_candidate['path'])
        if not error and len(minified_src.encode('latin-1')) < len(src_content.encode('latin-1')):
            best_text = minified_src
        else:
            best_text = src_content
        
        # Try compression
        candidate_bytes = best_text.encode('latin-1')
        compressed_bytes = zip_src(candidate_bytes)
        
        # Use the smaller version
        if len(compressed_bytes) < len(candidate_bytes):
            final_content = compressed_bytes
        else:
            final_content = candidate_bytes
        
        with open(path_out, "wb") as task_out:
            task_out.write(final_content)

print(f"\nResults:")
print(f"Scripts already compressed (skipped optimization): {already_compressed_count}")
print(f"Minification saved: {minify_save}b across all tasks")
print(f"Minification errors: {minify_errors}")

# Show detailed minification error information
if minify_error_details:
    print(f"\nDetailed Minification Errors:")
    for i, error_detail in enumerate(minify_error_details, 1):
        print(f"  {i}. {error_detail}")

print(f"Zlib compression saved: {total_save}b")
print(f"Projected score: {score}")

# Report default source verification results
print(f"\nDefault Source Verification Summary:")
print(f"Default source: {default_source}")
print(f"Default verifications attempted: {default_verification_attempted}")
print(f"Default verifications failed: {default_verification_failed}")
if default_verification_attempted > 0:
    default_success_rate = ((default_verification_attempted - default_verification_failed) / default_verification_attempted) * 100
    print(f"Default verification success rate: {default_success_rate:.1f}%")

if failed_default_tasks:
    print(f"\nFailed Default Source Tasks ({len(failed_default_tasks)} tasks):")
    for failed_task in failed_default_tasks:
        print(f"  Task {failed_task['task_num']:03d}: {failed_task['source']}")
        print(f"    Error: {failed_task['error'][:100]}...")  # Truncate long error messages

# Report non-default source usage
print(f"\nNon-Default Source Usage:")
print(f"Tasks using non-default sources: {len(non_default_tasks)}")

# Clarify what sizes are being measured
print(f"\nSize Measurement Clarification:")
print(f"  - Non-default sources were chosen because their FINAL OPTIMIZED size was smaller")
print(f"  - CSV shows all optimization stages: raw → minified → compressed")
print(f"  - Empty columns mean that optimization stage wasn't beneficial")
print(f"  - 'final_bytes_saved' = default_final_size - chosen_final_size")
print(f"  - 'Total bytes saved' shows additional optimization savings (0 for already-compressed scripts)")
print(f"  - Already-compressed scripts contribute by providing better solutions, not additional savings")

if non_default_tasks:
    print(f"\nNon-default source details:")
    for task_num, task_info in sorted(non_default_tasks.items()):
        print(f"  Task {task_num}: {task_info['source']}")

# Show source usage statistics
print(f"\nSource Usage Statistics:")
for source, stats in source_stats.items():
    if stats['tasks_used'] > 0:
        avg_original = stats['original_bytes'] / stats['tasks_used']
        avg_final = stats['final_bytes'] / stats['tasks_used']
        avg_savings = stats['bytes_saved'] / stats['tasks_used']
        is_default = " (DEFAULT)" if source == default_source else ""
        
        # Get list of tasks that used this source
        tasks_from_source = [task_num for task_num, task_source in task_source_map.items() if task_source == source]
        tasks_from_source.sort()
        
        print(f"Source: {source}{is_default}")
        print(f"  Tasks used: {stats['tasks_used']}")
        print(f"  Task IDs: {tasks_from_source}")
        if stats['already_compressed_tasks'] > 0:
            print(f"  Already compressed tasks: {stats['already_compressed_tasks']}")
        print(f"  Total bytes saved: {stats['bytes_saved']:,}")
        print(f"  Average original size: {avg_original:.1f} bytes")
        print(f"  Average final size: {avg_final:.1f} bytes")
        print(f"  Average savings per task: {avg_savings:.1f} bytes")
        print()
```

Additional analysis and final zip creation

```python
def analyze_compression_effectiveness():
    """Analyze which tasks benefit most from compression"""
    compression_stats = []
    
    for task_num in range(1, 401):
        task_path = f"{submission}/task{task_num:03d}.py"
        if os.path.exists(task_path):
            with open(task_path, "rb") as f:
                content = f.read()
                
            # Check if it's a compressed solution
            if content.startswith(b"#coding:L1"):
                compression_stats.append({
                    'task': task_num,
                    'size': len(content),
                    'compressed': True
                })
            else:
                compression_stats.append({
                    'task': task_num,
                    'size': len(content),
                    'compressed': False
                })
    
    compressed_count = sum(1 for s in compression_stats if s['compressed'])
    total_tasks = len(compression_stats)
    
    print(f"\nCompression Analysis:")
    print(f"Tasks using zlib compression: {compressed_count}/{total_tasks}")
    print(f"Average size of compressed solutions: {sum(s['size'] for s in compression_stats if s['compressed']) / max(1, compressed_count):.1f} bytes")
    print(f"Average size of uncompressed solutions: {sum(s['size'] for s in compression_stats if not s['compressed']) / max(1, total_tasks - compressed_count):.1f} bytes")
    
    return compression_stats

def create_others_zip():
    """Create others.zip containing tasks that used non-default sources and CSV metadata"""
    if not non_default_tasks:
        print("No non-default sources used - skipping others.zip creation")
        return None
    
    if ENVIRONMENT == "kaggle":
        others_zip_path = "/kaggle/working/others.zip"
        csv_path = "/kaggle/working/others_metadata.csv"
    else:
        others_zip_path = "./others.zip"
        csv_path = "./others_metadata.csv"
    
    # Create CSV file with all optimization stages
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'task_number', 'source',
            'default_raw_size', 'default_minified_size', 'default_compressed_size', 'default_minify_error',
            'chosen_raw_size', 'chosen_minified_size', 'chosen_compressed_size', 'chosen_minify_error',
            'final_bytes_saved'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for task_num, task_info in sorted(non_default_tasks.items()):
            default = task_info['default_stages']
            current = task_info['current_stages']
            
            writer.writerow({
                'task_number': f"task{task_num:03d}",
                'source': task_info['source'],
                'default_raw_size': default['raw_size'] if default['raw_size'] > 0 else '',
                'default_minified_size': default['minified_size'] if default['minified_size'] > 0 else '',
                'default_compressed_size': default['compressed_size'] if default['compressed_size'] > 0 else '',
                'default_minify_error': default['minify_error'] if default['minify_error'] else '',
                'chosen_raw_size': current['raw_size'],
                'chosen_minified_size': current['minified_size'] if current['minified_size'] > 0 else '',
                'chosen_compressed_size': current['compressed_size'],
                'chosen_minify_error': current['minify_error'] if current['minify_error'] else '',
                'final_bytes_saved': task_info['final_bytes_saved']
            })
    
    # Create zip file with just the original source files
    with zipfile.ZipFile(others_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for task_num, task_info in sorted(non_default_tasks.items()):
            # Add only the original source file (no metadata txt files)
            task_filename = f"task{task_num:03d}.py"
            zipf.writestr(task_filename, task_info['content'])
    
    zip_size = os.path.getsize(others_zip_path)
    csv_size = os.path.getsize(csv_path)
    
    print(f"\nCreated others.zip and metadata:")
    print(f"  Others.zip path: {others_zip_path}")
    print(f"  Others.zip size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")
    print(f"  CSV metadata path: {csv_path}")
    print(f"  CSV size: {csv_size:,} bytes")
    print(f"  Tasks from non-default sources: {len(non_default_tasks)}")
    
    # Show some statistics from the CSV data
    total_bytes_saved = sum(task_info['final_bytes_saved'] for task_info in non_default_tasks.values())
    avg_bytes_saved = total_bytes_saved / len(non_default_tasks) if non_default_tasks else 0
    print(f"  Total final bytes saved by using non-default sources: {total_bytes_saved:,}")
    print(f"  Average final bytes saved per task: {avg_bytes_saved:.1f}")
    
    return others_zip_path

def create_final_submission():
    """Create final submission with detailed reporting"""
    submission_zip = f"{submission}.zip"
    
    with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        task_count = 0
        total_size = 0
        
        for task_num in range(1, 401):
            task_id = f"{task_num:03d}"
            src_path = f"{submission}/task{task_id}.py"
            
            if os.path.exists(src_path):
                file_size = os.path.getsize(src_path)
                zipf.write(src_path, arcname=f"task{task_id}.py")
                task_count += 1
                total_size += file_size
    
    zip_size = os.path.getsize(submission_zip)
    
    print(f"\nFinal Submission:")
    print(f"Created submission zip with {task_count} tasks")
    print(f"Total uncompressed size: {total_size:,} bytes")
    print(f"Submission zip size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")
    print(f"Zip compression ratio: {zip_size/total_size:.3f}")
    
    return submission_zip

# Analyze compression effectiveness
compression_stats = analyze_compression_effectiveness()

# Create others.zip for non-default source tasks
others_zip = create_others_zip()

# Create final submission
final_zip = create_final_submission()
```

Summary Report

```python
def generate_summary_report():
    """Generate comprehensive summary of the optimization process"""
    print("="*60)
    print("="*60)
    
    print(f"Environment: {ENVIRONMENT.upper()}")
    print(f"Sources: {len(sources)} datasets/directories")
    print(f"Total Tasks: 400")
    
    print(f"\nOptimization Results:")
    print(f"Scripts already compressed (skipped): {already_compressed_count}")
    print(f"Minification saved: {minify_save:,} bytes")
    print(f"Minification errors: {minify_errors}")
    if minify_error_details:
        print(f"Error details: {len(minify_error_details)} files had minification issues")
    print(f"Zlib compression saved: {total_save:,} bytes")
    print(f"Total optimization: {minify_save + total_save:,} bytes")
    
    print(f"\nSolution Verification Results:")
    print(f"Total verifications attempted: {total_verification_attempted:,}")
    print(f"Verifications failed: {total_verification_failed:,}")
    print(f"Tasks with verification issues: {tasks_with_verification_issues}")
    if total_verification_attempted > 0:
        success_rate = ((total_verification_attempted - total_verification_failed) / total_verification_attempted) * 100
        print(f"Verification success rate: {success_rate:.1f}%")
    
    print(f"\nDefault Source Verification:")
    print(f"Default verifications attempted: {default_verification_attempted:,}")
    print(f"Default verifications failed: {default_verification_failed:,}")
    if default_verification_attempted > 0:
        default_success_rate = ((default_verification_attempted - default_verification_failed) / default_verification_attempted) * 100
        print(f"Default verification success rate: {default_success_rate:.1f}%")
    
    if failed_default_tasks:
        print(f"\nFailed Default Source Tasks ({len(failed_default_tasks)} tasks):")
        for failed_task in failed_default_tasks:
            print(f"  Task {failed_task['task_num']:03d}: {failed_task['source']}")
            print(f"    Error: {failed_task['error']}")
    else:
        print(f"\nAll default source tasks passed verification! ✓")
    
    print(f"Projected score: {score}")
    
    print(f"\nSource Performance Summary:")
    print(f"Default Source: {default_source}")
    for i, source in enumerate(sources):
        stats = source_stats.get(source, {'tasks_used': 0, 'bytes_saved': 0})
        is_default = " (DEFAULT)" if source == default_source else ""
        print(f"  {i}: {source}{is_default}")
        print(f"     Tasks contributed: {stats['tasks_used']}")
        print(f"     Total bytes saved: {stats['bytes_saved']:,}")
    
    # Show which source was most effective
    if source_stats:
        best_source = max(source_stats.keys(), key=lambda x: source_stats[x]['tasks_used'])
        best_stats = source_stats[best_source]
        print(f"\nMost Used Source: {best_source}")
        print(f"  Contributed to {best_stats['tasks_used']} tasks")
        print(f"  Total savings: {best_stats['bytes_saved']:,} bytes")
    
    # Report non-default source usage
    print(f"\nNon-Default Source Usage:")
    print(f"  Tasks using non-default sources: {len(non_default_tasks)}")
    if non_default_tasks:
        print(f"  Others.zip created with {len(non_default_tasks)} tasks")
        if others_zip:
            print(f"  Others.zip path: {others_zip}")
    else:
        print(f"  All tasks used the default source")
    
    print("="*60)

generate_summary_report()
print(f"Projected score: {score}")
```