# prompt-auditor
A three-pass AI agent that audits prompts for production failure risks. It detects ambiguity across 5 categories, rewrites the prompt in an attempt to avoid the detected ambiguities, then re-audits the rewrite to verify the issues are resolved. Uses the Anthropic API.
