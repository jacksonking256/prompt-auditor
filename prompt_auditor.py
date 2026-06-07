# ============================================================
# PROMPT AUDITOR AGENT
# ============================================================
# What this script does:
#   You give it a prompt written for an AI agent.
#   It runs that prompt through three automated passes:
#
#   Pass 1 - AUDIT:    Sends the prompt to Claude and asks it
#                      to find ambiguity problems across 5 categories.
#                      Scores the prompt on a 0-100 confidence scale.
#
#   Pass 2 - REWRITE:  Sends the original prompt and the audit findings
#                      back to Claude and asks it to produce a hardened
#                      rewrite that resolves every issue found.
#
#   Pass 3 - VERIFY:   Runs the rewritten prompt through the same audit
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


# ---- Imports ----------------------------------------------------------------
# Python's equivalent of "using" statements in C# or "import" in TypeScript.
# Each line pulls in a library so we can use its functionality.

import os           # Reads environment variables, including the API key
import anthropic    # Official Anthropic Python SDK for calling Claude
import csv          # Built-in library for reading and writing CSV files
import sys          # Provides access to command-line internals, used for sys.exit()
import argparse     # Built-in library for parsing command-line flags like --prompt and --file

from datetime import datetime       # Used to generate a timestamp for the output filename
from dotenv import load_dotenv      # Reads a .env file and loads its contents into the
                                    # environment so we don't have to set variables manually
                                    # each session. See .env.example for setup instructions.

# load_dotenv() reads the .env file in the current directory and injects its
# key-value pairs into os.environ. It must be called before anthropic.Anthropic()
# so the client can find ANTHROPIC_API_KEY. If no .env file exists, it does
# nothing and falls back to whatever variables are already set in the shell.
load_dotenv()


# ---- Constants --------------------------------------------------------------
# Python has no "const" keyword like C# or TypeScript. By convention, variables
# written in ALL_CAPS signal to other developers that they should not be
# reassigned. Python will not enforce this; it is a naming agreement.

MODEL = "claude-sonnet-4-6"
# Storing the model name as a constant means you only need to change it in one
# place if Anthropic releases a newer version later.


# ---- System Prompts ---------------------------------------------------------
# A system prompt is a block of instructions sent to Claude before the
# conversation starts. Think of it as the job description you hand to an
# employee before they begin work. The user's actual message comes separately.
# The system prompt establishes the role Claude plays for the entire session.
#
# This script uses two system prompts:
#   SYSTEM_PROMPT         - defines Claude's role as a prompt auditor
#   REWRITE_SYSTEM_PROMPT - defines Claude's role as a prompt hardening specialist
#
# Triple-quoted strings like """...""" are Python's syntax for multi-line strings.
# They are equivalent to backtick template literals in TypeScript.

SYSTEM_PROMPT = """You are a prompt auditor. Your job is to analyze AI agent prompts for ambiguity issues that could cause silent failures, hallucinations, or incorrect behavior.

You check for exactly five categories of issues:

1. SCOPE_LEAKAGE - Sequential instructions where logic from one step could bleed into another and act on the wrong items.
2. SUBJECTIVE_JARGON - Vague terms an AI cannot calculate without hallucinating (e.g., "the best one", "whichever has more validation behind it", "significant").
3. LITERAL_TRAPS - String matching vulnerabilities where casing, punctuation, or formatting variations could cause silent failures.
4. STRUCTURAL_FORMATTING - Formatting commands written as prose that an AI might print literally instead of apply.
5. SOURCE_DATA_ISSUES - Logical paradoxes, missing metadata, or context drift in the input data the prompt references.

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
# The rigid output format above is intentional. The parse_report() function
# below reads that exact structure. If Claude deviates from the format, parsing
# breaks. Think of the system prompt as a contract between this script and the model.


REWRITE_SYSTEM_PROMPT = """You are a prompt hardening specialist. You will be given an original AI agent prompt and a structured audit report listing its ambiguity issues.

Your job is to produce a single rewritten version of the prompt that resolves every finding in the audit report.

Rules:
- Preserve the original intent of the prompt exactly
- Address every finding listed - do not skip any
- Do not add unnecessary complexity or scope beyond what is needed to fix the issues
- Output only the rewritten prompt text, no explanation, no preamble, no labels
- Do not wrap the output in quotes or code blocks"""
# The "output only the rewritten prompt text" rule matters because if Claude
# adds a preamble like "Here is the rewritten prompt:", that text would end up
# in the CSV as part of the prompt itself. We also call .strip() on the response
# as a backup, but keeping the instruction explicit prevents the problem upstream.


# ---- API Client Setup -------------------------------------------------------
# This creates a single Anthropic API client that the entire script shares.
# The client reads ANTHROPIC_API_KEY from the environment automatically.
# That value is populated either by load_dotenv() reading your .env file,
# or by a variable you set manually in your shell session.
#
# Never hardcode your API key directly in source code. If you commit a real key
# to a public repo, automated bots will find it within minutes and use it.

client = anthropic.Anthropic()


# ---- Input Handling ---------------------------------------------------------

def get_input_prompt() -> str:
    """
    Determines what prompt the user wants to audit.

    Accepts input two ways:
      --prompt "text here"      The prompt is passed directly on the command line
      --file path/to/file.txt   The prompt is read from a text file on disk

    If neither flag is provided, this function prints usage instructions and exits.

    The -> str notation is a type hint. It tells readers and IDEs that this
    function returns a string. Python does not enforce type hints at runtime;
    they exist purely as documentation baked into the function signature.
    """

    # argparse is Python's standard library for building command-line interfaces.
    # It handles flag parsing, generates --help text automatically, and produces
    # readable error messages when the user passes something unexpected.
    parser = argparse.ArgumentParser(
        description="Audit an AI agent prompt for ambiguity issues."
    )

    # Each add_argument() call registers one flag the user can pass.
    # dest= sets the attribute name used to access the value after parsing.
    # help= is the description shown when the user runs --help.
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

    # parse_args() reads sys.argv (the actual command-line input) and returns
    # an object where each registered flag becomes an attribute.
    # For example: args.prompt, args.file
    args = parser.parse_args()

    if args.file:
        # "with open(...) as f" is Python's equivalent of a using() block in C#.
        # It guarantees the file is closed when the block exits, even if an
        # exception is thrown inside it.
        # "r" opens the file for reading only.
        # encoding="utf-8" prevents problems on Windows, where the default
        # encoding is sometimes cp1252 and can misread certain characters.
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read().strip()
            # .strip() removes any leading or trailing whitespace and newlines
            # from the file contents. This is standard cleanup when reading
            # text files, since editors often add a trailing newline.

    if args.prompt:
        return args.prompt

    # If we reach this point, the user did not pass either flag.
    # parser.print_help() writes the usage instructions to the console.
    # sys.exit(1) terminates the script and signals failure to the shell.
    # This is the same concept as returning a non-zero exit code in C++.
    parser.print_help()
    sys.exit(1)


# ---- Core API Calls ---------------------------------------------------------

def audit_prompt(user_prompt: str) -> str:
    """
    Sends a prompt to Claude for ambiguity analysis and returns the raw response.

    This function is used for both Pass 1 and Pass 3. The same audit logic
    runs on the original prompt and again on the rewritten prompt to verify
    that the issues were actually resolved.

    The return value is Claude's raw text response. It gets passed to
    parse_report() to convert it into structured data.
    """

    # client.messages.create() is the core API call.
    # We are building a single-turn conversation: one user message, one response.
    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,    # Sets the maximum length of Claude's response in tokens.
                            # 2048 is sufficient for a full audit report. Only increase
                            # this if responses are getting cut off, since higher values
                            # cost more per call.
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                # f"..." is an f-string, Python's template literal syntax.
                # Any expression inside curly braces gets evaluated and inserted
                # into the string. This is equivalent to `${variable}` in TypeScript.
                #
                # Wrapping the prompt in <prompt> XML tags signals to Claude that
                # this content is the subject of analysis, not a direct instruction
                # for it to follow.
                "content": f"Please audit the following AI agent prompt:\n\n<prompt>\n{user_prompt}\n</prompt>"
            }
        ]
    )

    # message.content is a list of content blocks. Claude can theoretically return
    # multiple blocks of different types, but for a plain text response there is
    # always exactly one. We access index [0] and read its .text attribute.
    return message.content[0].text


def rewrite_prompt(original_prompt: str, raw_report: str) -> str:
    """
    Sends the original prompt and audit findings to Claude and returns a
    hardened rewrite that resolves every issue found.

    This function handles Pass 2.

    We pass raw_report as an unmodified string rather than the parsed dict
    because we want Claude to see the full audit context exactly as it was
    generated, including all problem descriptions and suggested rewrites.
    """

    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=REWRITE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                # Wrapping the expression in parentheses lets us split a long
                # string across multiple lines without a line continuation character.
                # Python automatically concatenates adjacent string literals
                # when they appear inside parentheses.
                "content": (
                    f"Original prompt:\n<prompt>\n{original_prompt}\n</prompt>\n\n"
                    f"Audit report:\n<audit>\n{raw_report}\n</audit>\n\n"
                    f"Produce the hardened rewrite now."
                )
            }
        ]
    )

    # .strip() removes any leading or trailing whitespace from Claude's response.
    # This guards against cases where Claude adds an extra newline at the start
    # or end of its output despite being instructed not to.
    return message.content[0].text.strip()


# ---- Report Parsing ---------------------------------------------------------

def parse_report(raw_report: str) -> dict:
    """
    Converts Claude's raw text audit report into a structured Python dictionary.

    Input example (raw_report):
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

    Output structure:
        {
            "findings": [ {category, severity, location, problem, rewrite}, ... ],
            "summary": "This prompt has several issues..."
        }

    Parsing into a dict lets the rest of the script count findings, compute
    scores, and write clean CSV rows without working against raw text.
    """

    findings = []   # Accumulates one dict per FINDING block
    summary = ""    # Stores the summary paragraph as a plain string

    # Split the full report on "---" to isolate each FINDING block.
    # str.split(separator) works the same as String.Split() in C#.
    # The final chunk after the last "---" will contain the SUMMARY section.
    chunks = raw_report.strip().split("---")

    for chunk in chunks:
        chunk = chunk.strip()   # Remove surrounding whitespace from each chunk

        if not chunk:
            # Empty chunks appear when "---" separators are adjacent or trailing.
            # In Python, "not chunk" evaluates to True for empty strings.
            continue

        # Handle the SUMMARY section
        if chunk.startswith("SUMMARY"):
            # splitlines() returns a list of individual lines, equivalent to
            # Split('\n') in C#. The first line is the word "SUMMARY" itself,
            # so we use a slice [1:] to skip it and rejoin everything after.
            # A slice like [1:] means "give me every element from index 1 onward."
            lines = chunk.splitlines()
            summary = "\n".join(lines[1:]).strip()
            continue

        # Handle each FINDING block
        if "FINDING" in chunk:
            finding = {}    # Start with an empty dict and populate it line by line

            for line in chunk.splitlines():
                line = line.strip()

                # Each field follows the pattern "Key: Value".
                # We call split(": ", 1) to split only on the first occurrence.
                # The second argument limits how many splits are performed.
                # Without it, a colon inside the value (like in a URL) would
                # break the value into unwanted extra pieces.
                # This is equivalent to String.Split(':', 2) in C#.
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

            # Only append the finding if we actually parsed at least one field.
            # In Python, a non-empty dict evaluates to True in a boolean context.
            if finding:
                findings.append(finding)

    return {"findings": findings, "summary": summary}


# ---- Scoring ----------------------------------------------------------------

def compute_score(report: dict) -> dict:
    """
    Calculates a 0-100 confidence score from the parsed findings list.

    Higher is better. A score of 100 means no issues were detected.
    A score of 0 means the prompt is critically broken.

    Scoring logic:
      - Each HIGH finding subtracts 10 points
      - Each MEDIUM finding subtracts 5 points
      - Each LOW finding subtracts 2 points
      - Every finding also subtracts a flat 2-point penalty regardless of severity
      - The total penalty is capped at 100 so the score never goes negative
      - Final score = 100 minus the total penalty

    The flat per-finding penalty exists to ensure that a prompt with more
    findings always scores lower than one with fewer findings, even when the
    severity mix shifts between two runs.

    Scores are calculated from the structured findings data, not generated by
    the model. Deterministic math produces consistent, comparable results
    across runs. Asking Claude to generate a score would not.
    """

    severity_weights = {"HIGH": 10, "MEDIUM": 5, "LOW": 2}
    PER_FINDING_PENALTY = 2     # Applied to every finding on top of its severity weight.
                                # This ensures finding count always affects the score.

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    # Walk through every finding and increment the matching severity counter.
    # finding.get("severity", "") is a safe dictionary read. If the key is missing
    # for any reason, it returns an empty string instead of raising a KeyError.
    # Calling .upper() normalizes the value in case Claude returns lowercase.
    for finding in report["findings"]:
        severity = finding.get("severity", "").upper()
        if severity in counts:
            counts[severity] += 1

    total_findings = len(report["findings"])    # len() is equivalent to .Length or .Count in C#

    raw_score = (
        counts["HIGH"]   * severity_weights["HIGH"]   +
        counts["MEDIUM"] * severity_weights["MEDIUM"] +
        counts["LOW"]    * severity_weights["LOW"]    +
        total_findings   * PER_FINDING_PENALTY
    )

    # min(a, b) returns the smaller of two values, equivalent to Math.Min() in C#.
    # Capping at 100 ensures the confidence score never goes negative.
    risk_score = min(raw_score, 100)
    confidence_score = 100 - risk_score

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


# ---- CSV Export -------------------------------------------------------------

def export_csv(report: dict, input_prompt: str, score: dict,
               rewritten_prompt: str, verification: dict, verification_score: dict) -> str:
    """
    Writes the full audit results to a timestamped CSV file.

    The file contains three labeled sections:
      1. Original Prompt Audit  - all findings, summary, score, and the original prompt
      2. Hardened Rewrite       - the Claude-generated improved version of the prompt
      3. Verification Audit     - re-audit results for the rewrite, with its own score

    Each run writes a new file so nothing gets overwritten.
    Returns the filename so main() can print it to the console.
    """

    # strftime() formats a datetime object into a string using format codes.
    # "%Y%m%d_%H%M%S" produces output like "20250607_143022".
    # Including the timestamp in the filename guarantees uniqueness per run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_{timestamp}.csv"

    # "with open(...) as f" is Python's resource management block, equivalent
    # to a using() statement in C#. The file is guaranteed to close when the
    # block exits, even if an exception is raised inside it.
    #
    # "w" opens the file for writing and creates it if it does not exist.
    # newline="" is required by Python's csv module on Windows. Without it,
    # the writer inserts an extra blank line between every row.
    # encoding="utf-8" ensures special characters like quotes and symbols
    # are preserved correctly across platforms.
    with open(filename, "w", newline="", encoding="utf-8") as f:

        # csv.writer handles all the formatting details: quoting fields that
        # contain commas, escaping special characters, and so on.
        writer = csv.writer(f)

        # Section 1: Original Audit
        writer.writerow(["=== ORIGINAL PROMPT AUDIT ==="])
        writer.writerow(["Section", "Severity", "Category", "Location", "Problem", "Rewrite"])

        # Write one row per finding.
        # .get("key", "") is a safe dictionary lookup. It returns the value if
        # the key exists, or an empty string if it does not. This avoids a
        # KeyError if a field was missing during parsing.
        for finding in report["findings"]:
            writer.writerow([
                "FINDING",
                finding.get("severity", ""),
                finding.get("category", ""),
                finding.get("location", ""),
                finding.get("problem", ""),
                finding.get("rewrite", ""),
            ])

        # The summary gets its own row with the other columns left empty
        writer.writerow(["SUMMARY", "", "", "", report.get("summary", ""), ""])

        writer.writerow([])     # Blank row acts as a visual separator when opened in Excel

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
        # Including the original prompt text keeps the CSV self-contained.
        # Anyone reading the file later can see exactly what was audited.
        writer.writerow(["AUDITED PROMPT", "", "", "", input_prompt, ""])

        # Section 2: Hardened Rewrite
        writer.writerow([])
        writer.writerow(["=== HARDENED REWRITE ==="])
        writer.writerow(["REWRITTEN PROMPT", "", "", "", rewritten_prompt, ""])

        # Section 3: Verification Audit
        writer.writerow([])
        writer.writerow(["=== VERIFICATION AUDIT (rewrite re-audited) ==="])
        writer.writerow(["Section", "Severity", "Category", "Location", "Problem", "Rewrite"])

        if verification["findings"]:
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


# ---- Main Entrypoint --------------------------------------------------------

def main():
    """
    Runs the full three-pass audit pipeline in sequence:
      1. Audit the original prompt
      2. Generate a hardened rewrite
      3. Verify the rewrite by re-auditing it
      4. Export all results to a CSV file
    """

    input_prompt = get_input_prompt()

    # Pass 1: Audit the original prompt
    print("Pass 1: Auditing original prompt...\n")

    raw = audit_prompt(input_prompt)        # Raw text response from Claude
    report = parse_report(raw)              # Structured dict of findings and summary
    score = compute_score(report)           # Numeric score derived from findings

    # enumerate(list, start=1) yields both the index and the item on each iteration.
    # This is equivalent to a foreach loop with a built-in counter.
    # The "i, finding" syntax is tuple unpacking. It is the same concept as
    # destructuring in TypeScript: const [i, finding] = ...
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

    # Pass 2: Generate a hardened rewrite
    print("Pass 2: Generating hardened rewrite...\n")

    # We pass the unparsed raw report so Claude sees the full context, including
    # every problem description and suggested rewrite from the audit.
    rewritten = rewrite_prompt(input_prompt, raw)

    print("Rewritten prompt:")
    print(rewritten)
    print()

    # Pass 3: Verify the rewrite
    print("Pass 3: Verifying rewrite...\n")

    # Re-running the same audit function on the rewritten prompt confirms whether
    # the issues were actually resolved or if any ambiguity remains.
    verification_raw = audit_prompt(rewritten)
    verification_report = parse_report(verification_raw)
    verification_score = compute_score(verification_report)

    if verification_report["findings"]:
        remaining = len(verification_report["findings"])
        print(f"Verification found {remaining} remaining issue(s):\n")

        for i, finding in enumerate(verification_report["findings"], start=1):
            print(f"  Finding {i}: [{finding['severity']}] {finding['category']}")
            print(f"    Location: {finding['location']}")
            print(f"    Problem:  {finding['problem']}")
            print()
    else:
        print("Verification clean. No ambiguity issues detected in the rewrite.")

    print()
    print(f"Verification score: {verification_score['confidence_score']}/100 ({verification_score['grade']})")
    print()

    filename = export_csv(
        report, input_prompt, score,
        rewritten, verification_report, verification_score
    )
    print(f"Full report saved to: {filename}")


# ---- Script Entry Point -----------------------------------------------------
# This block runs only when the file is executed directly from the command line.
# It does not run if another Python file imports this one as a module.
#
# This pattern separates executable behavior from reusable logic. If someone
# imports this file to use its functions elsewhere, main() will not fire
# automatically. In C++ terms, it is the difference between code inside main()
# and code in a standalone function that could be called from anywhere.

if __name__ == "__main__":
    main()
