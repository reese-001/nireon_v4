#!/usr/bin/env python3
"""
Concatenate, clean, and split source files.

Usage
-----
    python concat.py [--html] [--keywords keyword1,keyword2] <dir1> [<dir2> ...]
    python concat.py --llm [--include-placeholders] [--include-config] [--keywords keyword1,keyword2]

• If --html is given, the script scans for .html/.htm/.css/.js files;
  otherwise it scans for .py files (default behaviour).
• If no directory is supplied for general mode, the current working directory is used.
• If --llm is given, it specifically pulls LLM subsystem files.
    • --include-placeholders: Also includes bootstrap/bootstrap_helper/placeholders.py
    • --include-config: Also includes configs/default/llm_config.yaml
• If --keywords is given, only processes files that have ALL specified keywords in their # process [...] comment
• Output files are written to the CWD.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Set, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Keyword filtering functions
# ──────────────────────────────────────────────────────────────────────────────

def _extract_process_keywords(file_path: Path) -> Set[str]:
    """
    Extract keywords from the # process [...] comment at the top of a Python file.
    Returns empty set if no process comment is found or if file can't be read.
    """
    try:
        # Only check the first few lines for the process comment
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Only check first 10 lines
                    break
                
                stripped = line.strip()
                if stripped.startswith('# process [') and stripped.endswith(']'):
                    # Extract content between brackets
                    bracket_content = stripped[11:-1]  # Remove '# process [' and ']'
                    
                    # Split by comma and clean up whitespace
                    keywords = [kw.strip() for kw in bracket_content.split(',')]
                    
                    # Filter out empty keywords
                    keywords = [kw for kw in keywords if kw]
                    
                    return set(keywords)
        
        return set()  # No process comment found
        
    except Exception as e:
        print(f"⚠️  Error reading process keywords from {file_path}: {e}")
        return set()


def _file_matches_keywords(file_path: Path, required_keywords: Set[str]) -> bool:
    """
    Check if a file contains ALL required keywords in its process comment.
    If no required keywords are specified, returns True (no filtering).
    If file has no process comment and keywords are required, returns False.
    """
    if not required_keywords:
        return True  # No filtering requested
    
    # Only check Python files for process comments
    if file_path.suffix.lower() not in (".py", ".md"):
        return False  
    
    file_keywords = _extract_process_keywords(file_path)
    
    # File must contain ALL required keywords
    return required_keywords.issubset(file_keywords)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_docstrings(src: str) -> str:
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if the first statement is an expression and that expression is a string constant
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and hasattr(node.body[0].value, 'value') # For ast.Constant in Py3.8+
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
                # For Python < 3.8, ast.Str might be used instead of ast.Constant
                elif (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Str)
                        and isinstance(node.body[0].value.s, str)):
                     node.body.pop(0)
        return ast.unparse(tree)
    except Exception: # Broad exception if parsing/unparsing fails (e.g. on non-Python files)
        return src


def _filter_lines(code: str) -> str:
    kept = []
    for line in code.splitlines():
        s = line.strip()
        if not s or s.startswith("#"): # Keep file path headers
            if line.startswith("# C:\\Users\\") or line.startswith("# /mnt/"): # common path headers
                 kept.append(line)
            continue
        kept.append(line)
    # Ensure a single newline at the end if there's content
    return "\n".join(kept).strip() + "\n" if kept else ""


def _read_text(path: Path) -> str:
    """
    Try UTF-8 first, then fall back to latin-1 with replacement.
    This avoids hard-stops on odd characters such as © or smart quotes.
    """
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        print(f"⚠️  UnicodeDecodeError for {path}, trying latin-1.")
        return path.read_text(encoding="latin-1", errors="replace")
    except Exception as e:
        print(f"⚠️  Could not read {path}: {e}")
        return ""

def _process_source(src: str, path: Path) -> str: # Modified to take Path
    """
    • For .py  : strip docstrings + remove blank / comment lines.
    • For other: just normalise line endings; keep everything else untouched.
    """
    suffix = path.suffix.lower()
    if suffix == ".py":
        try:
            stripped_src = _strip_docstrings(src)
            return _filter_lines(stripped_src)
        except SyntaxError:
            print(f"⚠️  SyntaxError stripping docstrings from {path}, including raw.")
            # Fallback to raw processing if stripping fails for a .py file
            return src.replace("\r\n", "\n").rstrip() + "\n"
        except Exception as e:
            print(f"⚠️  Error processing Python source {path}: {e}, including raw.")
            return src.replace("\r\n", "\n").rstrip() + "\n"
    else: # For .yaml, .html, .css, .js, .txt etc.
        return src.replace("\r\n", "\n").rstrip() + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

EXCLUDED_EXACT   = {"concat.py"} # This script itself
EXCLUDED_PREFIX  = ("nireon_combined_",) # Skip previously generated outputs
OUTPUT_PREFIX    = "nireon_combined_"     # Used to skip walk into output dir

def _eligible_general(path: Path, allowed_exts: Set[str], required_keywords: Set[str] = None) -> bool:
    name = path.name
    if name in EXCLUDED_EXACT or name.startswith(EXCLUDED_PREFIX):
        return False
    
    # Check file extension
    if path.suffix.lower() not in allowed_exts:
        return False
    
    # Check keyword matching if keywords are specified
    if required_keywords:
        return _file_matches_keywords(path, required_keywords)
    
    return True

def _discover_files_general(root: Path, allowed_exts: Set[str], required_keywords: Set[str] = None) -> List[Path]:
    found = []
    for current_root, dirs, files in os.walk(root):
        # prune any generated output directories
        dirs[:] = [d for d in dirs if not d.startswith(OUTPUT_PREFIX)]
        for fname in files:
            p = Path(current_root, fname)
            if _eligible_general(p, allowed_exts, required_keywords):
                found.append(p.resolve())
    return found

def _discover_llm_files(project_root: Path, include_placeholders: bool, include_config: bool, required_keywords: Set[str] = None) -> List[Path]:
    llm_files = []
    print(f"🔍 Project root for LLM discovery: {project_root}")

    # 1. domain/ports/llm_port.py
    llm_port_path = project_root / "domain" / "ports" / "llm_port.py"
    if llm_port_path.exists():
        if not required_keywords or _file_matches_keywords(llm_port_path, required_keywords):
            llm_files.append(llm_port_path)
            print(f"  + Found: {llm_port_path.relative_to(project_root)}")
        else:
            print(f"  - Filtered out (keywords): {llm_port_path.relative_to(project_root)}")
    else:
        print(f"  - Missing: {llm_port_path.relative_to(project_root)}")


    # 2. All .py files in infrastructure/llm/
    infra_llm_dir = project_root / "infrastructure" / "llm"
    if infra_llm_dir.is_dir():
        for item_path in infra_llm_dir.rglob("*.py"): # rglob for recursive
            if item_path.is_file():
                if not required_keywords or _file_matches_keywords(item_path, required_keywords):
                    llm_files.append(item_path)
                    print(f"  + Found: {item_path.relative_to(project_root)}")
                else:
                    print(f"  - Filtered out (keywords): {item_path.relative_to(project_root)}")
    else:
        print(f"  - Directory not found: {infra_llm_dir.relative_to(project_root)}")


    # 3. Optional: bootstrap/bootstrap_helper/placeholders.py
    if include_placeholders:
        placeholders_path = project_root / "bootstrap" / "bootstrap_helper" / "placeholders.py"
        if placeholders_path.exists():
            if not required_keywords or _file_matches_keywords(placeholders_path, required_keywords):
                llm_files.append(placeholders_path)
                print(f"  + Found (optional): {placeholders_path.relative_to(project_root)}")
            else:
                print(f"  - Filtered out (keywords, optional): {placeholders_path.relative_to(project_root)}")
        else:
            print(f"  - Missing (optional): {placeholders_path.relative_to(project_root)}")

    # 4. Optional: configs/default/llm_config.yaml
    if include_config:
        llm_config_path = project_root / "configs" / "default" / "llm_config.yaml"
        if llm_config_path.exists():
            # YAML files are not filtered by keywords (they don't have process comments)
            # But if keywords are specified, we skip non-Python files
            if not required_keywords:
                llm_files.append(llm_config_path)
                print(f"  + Found (optional): {llm_config_path.relative_to(project_root)}")
            else:
                print(f"  - Filtered out (non-Python with keywords): {llm_config_path.relative_to(project_root)}")
        else:
            print(f"  - Missing (optional): {llm_config_path.relative_to(project_root)}")
            
    # Remove duplicates and sort
    unique_files = sorted(list(set(p.resolve() for p in llm_files)))
    return unique_files

# ──────────────────────────────────────────────────────────────────────────────
# Splitting logic
# ──────────────────────────────────────────────────────────────────────────────

def _split_code(code: str,
                base_name: str,
                max_lines: int = 20000, # Increased max lines as per typical LLM context
                marker_prefix: str = "# C:\\Users\\") -> List[tuple[str, str]]: # Made marker prefix more general
    """Return list of (filename, code_chunk) pairs."""
    lines              = code.splitlines()
    chunks             = []
    buf                = []
    line_count         = 0
    # Store index of the last line that starts with marker_prefix OR is a `--- START OF FILE`
    last_good_split_point_idx  = 0
    file_idx           = 1

    custom_marker = "--- START OF FILE"

    for i, line in enumerate(lines):
        buf.append(line)
        line_count += 1

        is_path_marker = line.strip().lower().startswith(marker_prefix.lower())
        is_custom_marker = line.strip().startswith(custom_marker)

        if is_path_marker or is_custom_marker:
            last_good_split_point_idx = len(buf) -1 # Current line is a good split point


        if line_count >= max_lines:
            # Prefer splitting at last_good_split_point_idx if it's within a reasonable lookback
            # (e.g., last 20% of lines or if it's not too far back)
            # If no good marker, split at current length.
            split_at = len(buf)
            if last_good_split_point_idx > 0 and (len(buf) - last_good_split_point_idx < 0.5 * max_lines): # Heuristic
                # Ensure we split *before* the marker line so it starts the new chunk
                split_at = last_good_split_point_idx
            
            # If split_at is 0 (e.g. first file header is the only marker and > max_lines)
            # or if the split point is the entire buffer (no useful marker found recently)
            # then split at max_lines to make progress.
            if split_at == 0 or split_at == len(buf):
                split_at = max_lines if len(buf) > max_lines else len(buf)


            chunk_lines = buf[:split_at]
            # Ensure the chunk ends with a newline
            chunk_content = "\n".join(chunk_lines).rstrip() + "\n"
            
            # Add "--- START OF FILE" marker only if not already the first line
            if not chunk_lines[0].strip().startswith(custom_marker):
                 chunk_content = f"{custom_marker} {base_name}_{file_idx}.txt ---\n" + chunk_content


            chunks.append((f"{base_name}_{file_idx}.txt", chunk_content))
            file_idx += 1
            buf       = buf[split_at:]
            line_count = len(buf)
            last_good_split_point_idx = 0 # Reset for the new buffer
            # If the new buffer starts with a marker, update last_good_split_point_idx
            if buf and (buf[0].strip().lower().startswith(marker_prefix.lower()) or \
                        buf[0].strip().startswith(custom_marker)):
                last_good_split_point_idx = 0


    if buf: # Remaining buffer
        chunk_lines = buf
        chunk_content = "\n".join(chunk_lines).rstrip() + "\n"
        if not chunk_lines[0].strip().startswith(custom_marker):
             chunk_content = f"{custom_marker} {base_name}_{file_idx}.txt ---\n" + chunk_content
        chunks.append((f"{base_name}_{file_idx}.txt", chunk_content))
    return chunks

# ──────────────────────────────────────────────────────────────────────────────
# Main Processing Function
# ──────────────────────────────────────────────────────────────────────────────

def process_and_write_files(
        files_to_process: List[Path],
        output_base_name: str,
        project_root_for_headers: Path
    ) -> None:

    if not files_to_process:
        print("⚠️  No eligible source files found to process.")
        return

    print(f"🔄  Processing {len(files_to_process)} files for output: {output_base_name} ...")

    combined_content_parts = []
    processed_count = 0
    for p in files_to_process:
        try:
            raw_content = _read_text(p)
            if not raw_content.strip() and p.suffix.lower() not in (".py", ".md"): # Skip empty non-python files
                print(f"ℹ️  Skipping empty file: {p.relative_to(project_root_for_headers)}")
                continue

            # For Python files, process them. For others (like YAML), include mostly as-is.
            cleaned_or_raw_content = _process_source(raw_content, p)

            try:
                # Use a consistent header format for all files
                relative_path_header = p.resolve().relative_to(project_root_for_headers.resolve())
            except ValueError: # If p is not under project_root_for_headers (shouldn't happen with resolve)
                relative_path_header = p.name

            # Standardized header for all files
            combined_content_parts.append(f"# {project_root_for_headers.name}{os.sep}{relative_path_header}")
            combined_content_parts.append(cleaned_or_raw_content) # Add newline if not already there
            if not cleaned_or_raw_content.endswith("\n"):
                 combined_content_parts.append("\n")

            processed_count += 1
        except Exception as exc:
            print(f"⚠️  Error processing file {p}: {exc}")

    if not combined_content_parts:
        print("⚠️ No content generated after processing files.")
        return

    # Join all parts. Each part (header, content) should manage its own newlines.
    full_combined_text = "\n".join(combined_content_parts)

    # Determine the marker prefix dynamically or use a robust one
    # This finds the common prefix of absolute paths on the system
    common_path_prefix = os.path.commonpath([str(f.parent.resolve()) for f in files_to_process if f.is_file()])
    # A more robust marker for splitting could be "# C:\Path\To\Project\" or similar
    # For now, let's stick to a generic start or the custom marker.
    # The _split_code function already handles "--- START OF FILE"
    
    split_files = _split_code(full_combined_text, output_base_name, marker_prefix="# ") # Generic marker prefix

    for fname, content in split_files:
        Path(fname).write_text(content, encoding="utf-8")
        print(f"✅  Wrote {fname}  ({content.count(chr(10))+1} lines)")

    print(f"🎉  Done: {processed_count} source files → {len(split_files)} output files.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate, clean, and split source files.")
    parser.add_argument(
        "--html", action="store_true",
        help="Process .html/.htm/.css/.js files instead of .py (general mode)"
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Specifically pull LLM subsystem files. Ignores <dirs> and --html."
    )
    parser.add_argument(
        "--include-placeholders", action="store_true",
        help="With --llm, also include bootstrap placeholders. (default: False)"
    )
    parser.add_argument(
        "--include-config", action="store_true",
        help="With --llm, also include default llm_config.yaml. (default: False)"
    )
    parser.add_argument(
        "--keywords", type=str,
        help="Comma-separated list of keywords. Only include files with ALL keywords in their # process [...] comment"
    )
    parser.add_argument(
        "dirs", nargs="*",
        help="Directories to scan for general mode (default: CWD if not --llm)"
    )
    args = parser.parse_args()

    # Parse keywords from comma-delimited string
    required_keywords = set()
    if args.keywords:
        keywords_list = [kw.strip() for kw in args.keywords.split(',')]
        required_keywords = set(kw for kw in keywords_list if kw)  # Remove empty strings
        print(f"🔍 Filtering files by keywords: {', '.join(sorted(required_keywords))}")

    # --- HARDCODED PROJECT ROOT ---
    project_root_str = r"C:\Users\erees\Documents\development\nireon_v4"
    project_root = Path(project_root_str)
    # --- END HARDCODED PROJECT ROOT ---

    # Verify the hardcoded path looks like a Nireon project root
    if not project_root.is_dir() or \
       not (project_root / "domain").is_dir() or \
       not (project_root / "infrastructure").is_dir():
        print(f"❌ Error: The hardcoded project root '{project_root_str}' does not appear to be a valid Nireon v4 project directory.")
        print(f"  Please ensure the path is correct and contains 'domain' and 'infrastructure' subdirectories.")
        sys.exit(1)
    print(f"ℹ️  Using hardcoded project root: {project_root}")


    if args.llm:
        print("🚀 LLM Subsystem Mode Activated 🚀")
        llm_files_to_process = _discover_llm_files(
            project_root,
            args.include_placeholders,
            args.include_config,
            required_keywords
        )
        output_base = f"{OUTPUT_PREFIX}llm_subsystem"
        if required_keywords:
            # Add keywords to filename for clarity
            keywords_str = "_".join(sorted(required_keywords))
            output_base = f"{OUTPUT_PREFIX}llm_subsystem_{keywords_str}"
        process_and_write_files(llm_files_to_process, output_base, project_root)
    else:
        print("📚 General Directory Processing Mode 📚")
        if args.html:
            exts_general = {".html", ".htm", ".css", ".js"}
            print("Processing HTML/CSS/JS files.")
        else:
            exts_general = {".py"}
            print("Processing Python files.")

        # For general mode, if no dirs are provided, default to CWD *relative to where script is run*
        # Or, if you always want it relative to the hardcoded project_root for general mode:
        # target_dirs_general = [Path(p) for p in args.dirs] if args.dirs else [project_root]
        target_dirs_general = args.dirs or [os.getcwd()] # Current behavior: default to CWD
        
        resolved_dirs = []
        for p_str in target_dirs_general:
            p = Path(p_str)
            if not p.is_absolute(): # If a relative path is given, resolve it against CWD
                p = Path.cwd() / p
            resolved_dirs.append(p.resolve())


        valid_dirs = [d for d in resolved_dirs if d.is_dir()]
        if not valid_dirs:
            print("❌  No valid directories supplied for general mode.")
            sys.exit(1)

        # Output base name for general mode can still be based on the first *valid target directory*
        # or you might want to base it on the project_root if no dirs are given and you default to project_root
        first_dir_name = valid_dirs[0].name if valid_dirs[0].name else "cwd"
        output_base_general = f"{OUTPUT_PREFIX}{first_dir_name}"
        if required_keywords:
            # Add keywords to filename for clarity
            keywords_str = "_".join(sorted(required_keywords))
            output_base_general = f"{OUTPUT_PREFIX}{first_dir_name}_{keywords_str}"

        all_files_general = []
        seen_files_general = set()
        for d_general in valid_dirs:
            for f_general in _discover_files_general(d_general, exts_general, required_keywords):
                if f_general not in seen_files_general:
                    seen_files_general.add(f_general)
                    all_files_general.append(f_general)
        
        all_files_general.sort()
        # When calling process_and_write_files, the project_root_for_headers should still be your
        # actual project root for consistent relative path headers.
        process_and_write_files(all_files_general, output_base_general, project_root)