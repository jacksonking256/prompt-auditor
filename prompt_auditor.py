# ============================================================
# PROMPT AUDITOR AGENT
# ============================================================
# What this script does:
#   You give it a prompt written for an AI agent.
#   It runs that prompt through three automated passes:
#
#   Pass 1 -- AUDIT:    Sends the prompt to Claude and asks it
#                      to find ambiguity problems across 5 categories.
#                      Scores the prompt on a 0-100 confidence scale.
#
#   Pass 2 -- REWRITE:  Sends the original prompt + the audit findings
#                      back to Claude and asks it to produce a hardened
#                      rewrite that resolves every issue found.
#
#   Pass 3 -- VERIFY:   Runs the rewritten prompt through the same audit
#                      to confirm the issues are actually gone.
#
#   All three passes are exported to a timestamped CSV file.
#
# How to run it:
#   python prompt_auditor.py --prompt "Your prompt text here"
#   python prompt_auditor.py --file path/to/your/prompt.txt
#
# Setup:
#   1. Copy .env.example to .env and add your Anthropic API key
#   2. pip install -r requirements.txt
# ============================================================


# ── Imports ──────────────────────────────────────────────────────────────────
# These are Python's equivalent of "using" statements in C# or "import" in
# TypeScript. They pull in libraries so we can use their functionality.

import os           # Lets us read environment variables (like the API key)
import anthropic    # The official Anthropic Python SDK -- talks to Claude's API
import csv          # Built-in library for reading/writing CSV files
import sys          # Gives us access to command-line internals (used for sys.exit)
import argparse     # Built-in library for parsing command-line arguments (--prompt, --file)

from datetime import datetime       # We use this to put a timestamp in the output filename
from dotenv import load_dotenv      # Loads variables from a .env file into the environment
                                    # so we don't have to set them manually every session.
                                    # See .env.example for setup instructions.

# load_dotenv() reads the .env file in the current directory and injects its
# contents into os.environ. Must be called before anthropic.Anthropic() so the
# client can find ANTHROPIC_API_KEY. If no .env file exists, it does nothing --
# falling back to whatever env vars are already set in the shell.
load_dotenv()


# ── Constants ────────────────────────────────────────────────────────────────
# In Python there's no "const" keyword like C# or TypeScript.
# By convention, variables written in ALL_CAPS are treated as constants --
# other developers know not to reassign them. Python won't stop you, it's
# just a naming agreement the community follows.

MODEL = "claude-sonnet-4-6"
# This is the specific Claude model we're calling. If Anthropic releases a
# newer model later, you'd swap this string out here and it applies everywhere.


# ── System Prompts ───────────────────────────────────────────────────────────
# A "system prompt" is a set of instructions sent to Claude before the
# conversation starts. Think of it as the job description you hand to an
# employee before they start working. The user's actual message comes
# separately -- the system prompt just establishes who Claude is and what
# rules it should follow for this entire session.
#
# We have two system prompts:
#   SYSTEM_PROMPT         -- turns Claude into a prompt auditor
#   REWRITE_SYSTEM_PROMPT -- turns Claude into a prompt hardening specialist
#
# Triple-quoted strings ("""...""") are Python's way of writing multi-line
# strings. Equivalent to a template literal in TypeScript (backtick strings).

SYSTEM_PROMPT = """You are a prompt auditor. Your job is to analyze AI agent prompts for ambiguity issues that could cause silent failures, hallucinations, or incorrect behavior.

You check for exactly five categories of issues:

1. SCOPE_LEAKAGE -- Sequential instructions where logic from one step could bleed into another and act on the wrong items.
2. SUBJECTIVE_JARGON -- Vague terms an AI cannot calculate without hallucinating (e.g., "the best one", "whichever has more validation behind it", "significant").
3. LITERAL_TRAPS -- String matching vulnerabilities where casing, punctuation, or formatting variations could cause silent failures.
4. STRUCTURAL_FORMATTING -- Formatting commands written as prose that an AI might print literally instead of apply.
5. SOURCE_DATA_ISSUES -- Logical paradoxes, missing metadata, or context drift in the input data the prompt references.

For each issue you find, output it in this exact format:

FINDING
Category: <category name from the list above>
Severity: <LOW | MEDIUM | HIGH>
Location: <quote the specific phrase or sentence that has the problem>
Problem: <one or two sentences explaining why this is ambiguous or dangerous>
Rewrite: <a concrete suggested fix for just that phrase or instruction>
---

After all findings, output a one-paragraph SUMMARY section labeled exactly as:

SUMMARY
<your paragraph here>

If you find no issues in a category, do not mention that category. If you find no issues at all, say so clearly."""
# NOTE: The rigid output format above (FINDING / Category: / --- / SUMMARY) is
# intentional. Our parse_report() function below reads that exact structure.
# If Claude deviates from the format, parsing breaks. The system prompt is
# essentially a contract between us and the model.


REWRITE_SYSTEM_PROMPT = """You are a prompt hardening specialist. You will be given an original AI agent prompt and a structured audit report listing its ambiguity issues.

Your job is to produce a single rewritten version of the prompt that resolves every finding in the audit report.

Rules:
- Preserve the original intent of the prompt exactly
- Address every finding listed -- do not skip any
- Do not add unnecessary complexity or scope beyond what is needed to fix the issues
- Output only the rewritten prompt text, no explanation, no preamble, no labels
- Do not wrap the output in quotes or code blocks"""
# NOTE: "Output only the rewritten prompt text" is important here. If Claude
# adds preamble like "Here is the rewritten prompt:", that extra text would
# end up in our CSV as part of the prompt. We strip() the response as a
# safety measure, but keeping the instruction clean prevents the problem.


# ── API Client Setup ─────────────────────────────────────────────────────────
# This creates a single Anthropic API client that the whole script shares.
# It reads ANTHROPIC_API_KEY from the environment automatically -- populated
# either by load_dotenv() above (from your .env file) or by a shell env var.
#
# Never hardcode your API key directly in source code. If you commit it to
# a public repo, bots will find it within minutes and drain your credits.

client = anthropic.Anthropic()


# ── Input Handling ───────────────────────────────────────────────────────────

def get_input_prompt() -> str:
    """
    Figures out what prompt the user wants to audit.

    Accepts input two ways:
      --prompt "text here"    The prompt is passed directly on the command line
      --file path/to/file.txt The prompt is read from a text file

    If neither is provided, prints usage instructions and exits.

    The -> str notation is a "type hint" -- it tells readers (and IDEs) that
    this function returns a string. Python doesn't enforce it at runtime,
    it's just documentation baked into the signature.
    """

    # argparse is Python's standard library for building CLI tools.
    # It handles --flag parsing, generates --help text, and gives us
    # clean error messages if the user passes something unexpected.
    parser = argparse.ArgumentParser(
        description="Audit an AI agent prompt for ambiguity issues."
    )

    # Each add_argument() call registers one flag the user can pass.
    # dest= is the variable name we'll use to access the value later.
    # help= is the text shown when the user runs --help.
    parser.add_argument(
        "--prompt",
        dest="prompt",
        help="The prompt text to audit, passed inline as a string."
    )
    parser.add_argument(
        "--file",
        dest="file",
        help="Path to a .txt file containing the prompt to audit."
    )

    # parse_args() reads sys.argv (the actual command-line arguments) and
    # returns an object where each flag becomes an attribute.
    # e.g. args.prompt, args.file
    args = parser.parse_args()

    if args.file:
        # "with open(...) as f" is Python's equivalent of a using() block in C#.
        # It guarantees the file is closed when the block exits, even if an
        # exception is thrown.
        # "r" = open for reading. encoding="utf-8" prevents issues on Windows
        # where the default encoding is sometimes cp1252 (which can't handle
        # some special characters).
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read().strip()
            # .strip() removes leading/trailing whitespace and newlines from
            # the file contents -- common cleanup step when reading text files.

    if args.prompt:
        return args.prompt

    # If we reach here, the user didn't pass either flag.
    # Print usage instructions and exit with a non-zero code.
    # sys.exit(1) signals to the shell that something went wrong --
    # same concept as returning a non-zero exit code in C++.
    parser.print_help()
    sys.exit(1)


# ── Core API Calls ───────────────────────────────────────────────────────────

def audit_prompt(user_prompt: str) -> str:
    """
    Sends a prompt to Claude for ambiguity analysis.

    This is Pass 1 and Pass 3 -- the same function is used for both
    the original audit and the verification audit on the rewrite.

    Returns the raw text response from Claude, which we'll parse
    into structured data in parse_report().
    """

    # client.messages.create() is the core API call.
    # We're building a one-turn conversation: one user message, one response.
    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,    # Maximum tokens Claude can use in its response.
                            # 2048 is plenty for an audit report. Raising this
                            # costs more -- only increase if responses get cut off.
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                # f"..." is an f-string -- Python's template literal syntax.
                # {variable} inside an f-string gets replaced with its value,
                # same as `${variable}` in TypeScript backtick strings.
                #
                # We wrap the prompt in <prompt> XML tags so Claude clearly
                # understands "this is the thing I'm analyzing" rather than
                # treating it as a direct instruction to follow.
                "content": f"Please audit the following AI agent prompt:\n\n<prompt>\n{user_prompt}\n</prompt>"
            }
        ]
    )

    # message.content is a list of content blocks (Claude can theoretically
    # return multiple blocks of different types). For a plain text response,
    # there's always exactly one block, and .text gives us its string content.
    return message.content[0].text


def rewrite_prompt(original_prompt: str, raw_report: str) -> str:
    """
    Sends the original prompt + audit findings to Claude and asks it to
    produce a hardened rewrite that resolves every issue.

    This is Pass 2.

    raw_report is the unmodified text string returned by audit_prompt() --
    we pass it directly rather than the parsed version so Claude sees the
    full context exactly as it was generated.
    """

    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=REWRITE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                # Parentheses around a multi-line string let us break it across
                # lines for readability without needing a line continuation
                # character. Python concatenates adjacent string literals
                # automatically inside parentheses.
                "content": (
                    f"Original prompt:\n<prompt>\n{original_prompt}\n</prompt>\n\n"
                    f"Audit report:\n<audit>\n{raw_report}\n</audit>\n\n"
                    f"Produce the hardened rewrite now."
                )
            }
        ]
    )

    # .strip() removes any leading/trailing whitespace from Claude's response.
    # This is a safety measure in case Claude adds a newline before or after
    # the rewritten prompt despite our instructions not to.
    return message.content[0].text.strip()


# ── Report Parsing ───────────────────────────────────────────────────────────

def parse_report(raw_report: str) -> dict:
    """
    Converts Claude's raw text audit report into a structured Python dictionary.

    Input (raw_report) looks like this:
        FINDING
        Category: SUBJECTIVE_JARGON
        Severity: HIGH
        Location: "cheapest game"
        Problem: "Cheapest" is undefined...
        Rewrite: Replace with explicit criteria...
        ---
        FINDING
        ...
        ---
        SUMMARY
        This prompt has several issues...

    Output is a dict with two keys:
        {
            "findings": [ {category, severity, location, problem, rewrite}, ... ],
            "summary": "This prompt has several issues..."
        }

    Why parse it? So the rest of the script can work with structured data
    instead of raw text -- count findings, compute scores, write clean CSV rows.
    """

    findings = []   # Will hold one dict per FINDING block
    summary = ""    # Will hold the summary paragraph text

    # Split the report on "---" to separate individual FINDING blocks.
    # str.split(separator) is like String.Split() in C#.
    # The last chunk after the final "---" will contain the SUMMARY section.
    chunks = raw_report.strip().split("---")

    for chunk in chunks:
        chunk = chunk.strip()   # Remove whitespace around each chunk

        if not chunk:
            # Skip empty chunks -- these appear when there are multiple
            # "---" separators in a row or at the end of the string.
            # "not chunk" is True for empty strings in Python.
            continue

        # ── Handle SUMMARY section ────────────────────────────────────
        if chunk.startswith("SUMMARY"):
            # splitlines() returns a list of lines -- same as Split('\n') in C#.
            # The first line is just the word "SUMMARY", so we skip it ([1:])
            # and rejoin the rest as the summary paragraph.
            # [1:] is a "slice" -- it means "give me everything from index 1
            # onward", skipping index 0 (the "SUMMARY" header line).
            lines = chunk.splitlines()
            summary = "\n".join(lines[1:]).strip()
            continue

        # ── Handle FINDING blocks ─────────────────────────────────────
        if "FINDING" in chunk:
            finding = {}    # Empty dict -- we'll populate it line by line

            for line in chunk.splitlines():
                line = line.strip()

                # Each field is formatted as "Key: Value".
                # split(": ", 1) splits on the FIRST occurrence of ": " only.
                # The maxsplit=1 argument is critical -- without it, a colon
                # inside the value (e.g. in a URL or example) would cause the
                # split to break the value into unwanted pieces.
                # This is like String.Split(':', 2) in C# -- "split at most once".
                if line.startswith("Category:"):
                    finding["category"] = line.split(": ", 1)[1]
                elif line.startswith("Severity:"):
                    finding["severity"] = line.split(": ", 1)[1]
                elif line.startswith("Location:"):
                    finding["location"] = line.split(": ", 1)[1]
                elif line.startswith("Problem:"):
                    finding["problem"] = line.split(": ", 1)[1]
                elif line.startswith("Rewrite:"):
                    finding["rewrite"] = line.split(": ", 1)[1]

            # Only add to the list if we actually parsed something.
            # "if finding" is True for a non-empty dict, False for an empty one.
            if finding:
                findings.append(finding)

    return {"findings": findings, "summary": summary}


# ── Scoring ──────────────────────────────────────────────────────────────────

def compute_score(report: dict) -> dict:
    """
    Calculates a confidence score (0-100) from the parsed findings.

    Scoring logic:
      - Each HIGH finding subtracts 10 points
      - Each MEDIUM finding subtracts 5 points
      - Each LOW finding subtracts 2 points
      - Each finding (any severity) subtracts an additional 2 points flat
        -- this ensures more findings always produce a lower score, even if
        the severity mix shifts toward lower weights between two runs
      - Raw penalty is capped at 100
      - Final score = 100 - penalty (100 = clean, 0 = catastrophically broken)

    We intentionally do NOT ask Claude to score the prompt -- deterministic
    math on structured data is more reliable and consistent across runs than
    asking an LLM to generate a number.
    """

    severity_weights = {"HIGH": 10, "MEDIUM": 5, "LOW": 2}
    PER_FINDING_PENALTY = 2     # Flat penalty per finding regardless of severity.
                                # Ensures that more findings always = lower score,
                                # even if the severity mix shifts between two runs.

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    # Loop through every finding and increment the appropriate counter.
    # finding.get("severity", "") safely reads the "severity" key --
    # if it's missing for some reason, it returns "" instead of crashing.
    # .upper() normalizes casing in case Claude returns "high" instead of "HIGH".
    for finding in report["findings"]:
        severity = finding.get("severity", "").upper()
        if severity in counts:
            counts[severity] += 1   # += is the same as in C# -- increment by 1

    total_findings = len(report["findings"])    # len() = .Length / .Count in C#

    # Calculate the total penalty score
    raw_score = (
        counts["HIGH"]   * severity_weights["HIGH"]   +
        counts["MEDIUM"] * severity_weights["MEDIUM"] +
        counts["LOW"]    * severity_weights["LOW"]    +
        total_findings   * PER_FINDING_PENALTY
    )

    # min(a, b) returns the smaller of two values -- same as Math.Min() in C#.
    # We cap the penalty at 100 so the confidence score never goes negative.
    risk_score = min(raw_score, 100)
    confidence_score = 100 - risk_score

    # Assign a plain-English grade band based on the final score
    if confidence_score >= 80:
        grade = "LOW RISK"
    elif confidence_score >= 50:
        grade = "MODERATE RISK"
    elif confidence_score >= 25:
        grade = "HIGH RISK"
    else:
        grade = "CRITICAL"

    return {
        "high_count":       counts["HIGH"],
        "medium_count":     counts["MEDIUM"],
        "low_count":        counts["LOW"],
        "total_findings":   total_findings,
        "confidence_score": confidence_score,
        "grade":            grade,
    }


# ── CSV Export ───────────────────────────────────────────────────────────────

def export_csv(report: dict, input_prompt: str, score: dict,
               rewritten_prompt: str, verification: dict, verification_score: dict) -> str:
    """
    Writes the complete audit results to a CSV file with three labeled sections:
      1. Original Prompt Audit  -- findings, summary, score, original prompt text
      2. Hardened Rewrite       -- the Claude-generated improved prompt
      3. Verification Audit     -- re-audit of the rewrite, with its own score

    The filename is timestamped so each run produces a new file and nothing
    gets overwritten.

    Returns the filename so main() can print it to the console.
    """

    # strftime() formats a datetime object as a string.
    # "%Y%m%d_%H%M%S" produces something like "20250607_143022"
    # This timestamp makes filenames unique per run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_{timestamp}.csv"

    # "with open(...) as f" opens the file and guarantees it's closed when
    # the block ends -- Python's equivalent of a using() block in C#.
    # "w" = write mode (creates the file, or overwrites it if it exists)
    # newline="" is required by Python's csv module on Windows -- without it,
    # the csv writer adds an extra blank line between every row.
    # encoding="utf-8" ensures special characters (quotes, dashes, symbols)
    # are preserved correctly.
    with open(filename, "w", newline="", encoding="utf-8") as f:

        # csv.writer wraps the file object and handles all CSV formatting
        # for us -- quoting fields that contain commas, escaping special chars, etc.
        writer = csv.writer(f)

        # ── Section 1: Original Audit ─────────────────────────────────
        writer.writerow(["=== ORIGINAL PROMPT AUDIT ==="])
        writer.writerow(["Section", "Severity", "Category", "Location", "Problem", "Rewrite"])

        # Loop through each finding and write one row per finding.
        # .get("key", "") is a safe dictionary lookup -- returns the value if
        # the key exists, or an empty string if it doesn't. Safer than
        # finding["key"] which would throw a KeyError if the field was missing.
        for finding in report["findings"]:
            writer.writerow([
                "FINDING",
                finding.get("severity", ""),
                finding.get("category", ""),
                finding.get("location", ""),
                finding.get("problem", ""),
                finding.get("rewrite", ""),
            ])

        # Summary row -- no severity/category/location, just the paragraph text
        writer.writerow(["SUMMARY", "", "", "", report.get("summary", ""), ""])

        # Blank row as a visual separator in Excel
        writer.writerow([])

        # Score header + data row
        writer.writerow([
            "SCORE SUMMARY",
            "High Findings", "Medium Findings", "Low Findings",
            "Total Findings", "Confidence Score", "Grade"
        ])
        writer.writerow([
            "",
            score["high_count"],
            score["medium_count"],
            score["low_count"],
            score["total_findings"],
            f"{score['confidence_score']}/100",
            score["grade"],
        ])

        writer.writerow([])
        # The original prompt text for traceability -- so the CSV is self-contained
        writer.writerow(["AUDITED PROMPT", "", "", "", input_prompt, ""])

        # ── Section 2: Hardened Rewrite ───────────────────────────────
        writer.writerow([])
        writer.writerow(["=== HARDENED REWRITE ==="])
        writer.writerow(["REWRITTEN PROMPT", "", "", "", rewritten_prompt, ""])

        # ── Section 3: Verification Audit ─────────────────────────────
        writer.writerow([])
        writer.writerow(["=== VERIFICATION AUDIT (rewrite re-audited) ==="])
        writer.writerow(["Section", "Severity", "Category", "Location", "Problem", "Rewrite"])

        if verification["findings"]:
            # Residual issues were found in the rewrite -- list them
            for finding in verification["findings"]:
                writer.writerow([
                    "FINDING",
                    finding.get("severity", ""),
                    finding.get("category", ""),
                    finding.get("location", ""),
                    finding.get("problem", ""),
                    finding.get("rewrite", ""),
                ])
        else:
            # Clean pass -- no issues detected in the rewritten prompt
            writer.writerow(["No findings detected in rewritten prompt.", "", "", "", "", ""])

        writer.writerow(["SUMMARY", "", "", "", verification.get("summary", ""), ""])
        writer.writerow([])
        writer.writerow([
            "SCORE SUMMARY",
            "High Findings", "Medium Findings", "Low Findings",
            "Total Findings", "Confidence Score", "Grade"
        ])
        writer.writerow([
            "",
            verification_score["high_count"],
            verification_score["medium_count"],
            verification_score["low_count"],
            verification_score["total_findings"],
            f"{verification_score['confidence_score']}/100",
            verification_score["grade"],
        ])

    return filename


# ── Main Entrypoint ──────────────────────────────────────────────────────────

def main():
    """
    Orchestrates the full three-pass audit pipeline:
      1. Audit the original prompt
      2. Generate a hardened rewrite
      3. Verify the rewrite by re-auditing it
      4. Export everything to CSV
    """

    # Step 0: Get the prompt to audit (from --prompt or --file flag)
    input_prompt = get_input_prompt()

    # ── Pass 1: Audit original prompt ────────────────────────────────
    print("Pass 1: Auditing original prompt...\n")

    # audit_prompt() calls the API and returns Claude's raw text response
    raw = audit_prompt(input_prompt)

    # parse_report() converts that raw text into a structured dict
    report = parse_report(raw)

    # compute_score() tallies up the findings into a numeric score
    score = compute_score(report)

    # Print each finding to the console.
    # enumerate(list, start=1) gives us both the index and the item in each
    # iteration -- like a foreach with a built-in counter.
    # "i, finding" is tuple unpacking -- same concept as destructuring in
    # TypeScript: const [i, finding] = ...
    for i, finding in enumerate(report["findings"], start=1):
        print(f"Finding {i}: [{finding['severity']}] {finding['category']}")
        print(f"  Location: {finding['location']}")
        print(f"  Problem:  {finding['problem']}")
        print(f"  Rewrite:  {finding['rewrite']}")
        print()

    print("SUMMARY")
    print(report["summary"])
    print()
    print(f"Score: {score['confidence_score']}/100 ({score['grade']})")
    print(f"  High: {score['high_count']}  Medium: {score['medium_count']}  Low: {score['low_count']}")
    print()

    # ── Pass 2: Generate hardened rewrite ────────────────────────────
    print("Pass 2: Generating hardened rewrite...\n")

    # We pass the raw (unparsed) report here so Claude sees the full
    # context -- every finding with its full problem description and
    # suggested rewrite -- rather than just the structured data subset.
    rewritten = rewrite_prompt(input_prompt, raw)

    print("Rewritten prompt:")
    print(rewritten)
    print()

    # ── Pass 3: Verify the rewrite ───────────────────────────────────
    print("Pass 3: Verifying rewrite...\n")

    # Re-run the exact same audit function on the rewritten prompt.
    # If the rewrite resolved all issues, findings should be empty or minimal.
    verification_raw = audit_prompt(rewritten)
    verification_report = parse_report(verification_raw)
    verification_score = compute_score(verification_report)

    if verification_report["findings"]:
        # Some issues remain -- print them so the user knows what's left
        remaining = len(verification_report["findings"])
        print(f"Verification found {remaining} remaining issue(s):\n")

        for i, finding in enumerate(verification_report["findings"], start=1):
            print(f"  Finding {i}: [{finding['severity']}] {finding['category']}")
            print(f"    Location: {finding['location']}")
            print(f"    Problem:  {finding['problem']}")
            print()
    else:
        # Clean pass -- the rewrite resolved everything the auditor could find
        print("Verification clean -- no ambiguity issues detected in rewrite.")

    print()
    print(f"Verification score: {verification_score['confidence_score']}/100 ({verification_score['grade']})")
    print()

    # ── Export to CSV ─────────────────────────────────────────────────
    # Pass all three passes worth of data into the CSV exporter.
    # The function returns the filename it created so we can tell the user.
    filename = export_csv(
        report, input_prompt, score,
        rewritten, verification_report, verification_score
    )
    print(f"Full report saved to: {filename}")


# ── Script Entry Point ───────────────────────────────────────────────────────
# This block only runs when you execute this file directly:
#   python prompt_auditor.py
#
# It does NOT run if another Python file imports this one as a module.
# This is a Python convention equivalent to checking if a file is the
# entry point vs. being used as a library.
#
# In C++ terms: this is like putting code in main() vs. a standalone function
# that could be called from anywhere. The check protects reusable code from
# running side effects on import.

if __name__ == "__main__":
    main()
