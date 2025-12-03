import os
import google.generativeai as genai
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None
import json
from PIL import Image
import base64
import glob
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
import re
import signal
import time
from functools import wraps
from multiprocessing import Process, Queue

# Optional providers
try:
    import anthropic  # Claude
except Exception:
    anthropic = None

try:
    from openai import OpenAI as OpenAIClient  # OpenAI GPT
except Exception:
    OpenAIClient = None


def _b64_image_from_path(image_path: Path) -> Tuple[str, str]:
    ext = image_path.suffix.lower().lstrip('.')
    mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else 'png'}"
    data = image_path.read_bytes()
    return mime, base64.b64encode(data).decode('utf-8')


class _GeminiVisionClient:
    def __init__(self, model_name: str, temperature: float = 0.1):
        self.model = genai.GenerativeModel(model_name, generation_config={"temperature": temperature})

    def infer(self, prompt: str, image_path: Path) -> str:
        try:
            img = Image.open(image_path)
            resp = self.model.generate_content([prompt, img])
        except Exception:
            # Fallback to raw bytes
            img_bytes = image_path.read_bytes()
            resp = self.model.generate_content([prompt, {"mime_type": "image/png", "data": img_bytes}])
        return getattr(resp, 'text', str(resp))


class _OpenAIVisionClient:
    def __init__(self, model_name: str, temperature: float = 0.1, base_url: Optional[str] = None):
        if OpenAIClient is None:
            raise RuntimeError("openai package not installed")
        # Prefer environment variable for key
        api_key = os.getenv('OPENAI_API_KEY') or OPENAI_API_KEY
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for OpenAI GPT")
        self.client = OpenAIClient(api_key=api_key, base_url=base_url) if base_url else OpenAIClient()
        self.model_name = model_name
        self.temperature = temperature

    def infer(self, prompt: str, image_path: Path) -> str:
        mime, b64 = _b64_image_from_path(image_path)
        messages = [
            {"role": "system", "content": "You are a meticulous circuit analysis expert. Follow the user instructions exactly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=messages,
        )
        return resp.choices[0].message.content


class _AnthropicVisionClient:
    def __init__(self, model_name: str, temperature: float = 0.1):
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")
        api_key = os.getenv('ANTHROPIC_API_KEY') or ANTHROPIC_API_KEY
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY for Anthropic Claude")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def infer(self, prompt: str, image_path: Path) -> str:
        mime, b64 = _b64_image_from_path(image_path)
        msg = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            temperature=self.temperature,
            system="You are a meticulous circuit analysis expert.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    ],
                }
            ],
        )
        parts = getattr(msg, 'content', [])
        texts = []
        for p in parts:
            if getattr(p, 'type', None) == 'text':
                texts.append(p.text)
        return "\n".join(texts) if texts else str(msg)


class _OpenRouterVisionClient:
    def __init__(self, model_name: str, temperature: float = 0.1, base_url: str = 'https://openrouter.ai/api/v1', api_key: Optional[str] = None, max_tokens: Optional[int] = None):
        if OpenAIClient is None:
            raise RuntimeError("openai package not installed")
        key = api_key or os.getenv('OPENROUTER_API_KEY') or OPENROUTER_API_KEY
        if not key:
            raise RuntimeError("Missing OPENROUTER_API_KEY for OpenRouter")
        self.client = OpenAIClient(api_key=key, base_url=base_url)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def infer(self, prompt: str, image_path: Path) -> str:
        mime, b64 = _b64_image_from_path(image_path)
        messages = [
            {"role": "system", "content": "You are a meticulous circuit analysis expert. Follow the user instructions exactly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ]
        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

# Load environment variables
load_dotenv()

# Configure the Gemini API
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or ''
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) for Google Gemini")
genai.configure(api_key=api_key)

# Additional API variables (read from environment by default)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')


# Model selection (default to route via RouteLLM). Override via env: SYMBOLIC_MODEL_PROVIDER / SYMBOLIC_MODEL_NAME
DEFAULT_MODEL_PROVIDER = os.getenv('SYMBOLIC_MODEL_PROVIDER', 'route')  # one of: route, gemini, claude, gpt
DEFAULT_MODEL_NAME = os.getenv('SYMBOLIC_MODEL_NAME', '')

# Configuration constants
STRATEGY_TIMEOUT_SECONDS = 5  # Hard limit per symbolic strategy
GLOBAL_COMPARE_TIMEOUT_SECONDS = 120  # Hard cap per question comparison
ROUTELLM_MAX_TOKENS = int(os.getenv('ROUTELLM_MAX_TOKENS', '512'))  # Output cap for RouteLLM

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the signal handler and a 5-second alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable the alarm
            return result
        return wrapper
    return decorator

class SymbolicEquationBenchmark:
    def __init__(self, dataset_path: str = "main_dataset/symbolic_equations_dataset_v13", init_model: bool = True, model_provider: str = DEFAULT_MODEL_PROVIDER, model_name: Optional[str] = DEFAULT_MODEL_NAME, temperature: float = 0.1):
        self.dataset_path = dataset_path
        self.model_provider = (model_provider or 'gemini').lower()
        self.model_name = model_name or ''
        self.temperature = temperature
        # Backward compatibility: keep self.model for gemini, but prefer client abstraction
        self.model = genai.GenerativeModel('gemini-2.5-pro-preview-06-05', generation_config={"temperature": temperature}) if (init_model and self.model_provider == 'gemini') else None
        self.client = self._build_client(self.model_provider, self.model_name, temperature) if init_model else None
        self.results = []

    def _is_html_error_text(self, text: str) -> bool:
        try:
            if not text:
                return False
            lowered = text.lower()
            if "<!doctype html" in lowered or "<html" in lowered:
                return True
            if lowered.startswith("error occurred during inference") and "<html" in lowered:
                return True
            if "cloudflare" in lowered and ("5xx" in lowered or "524" in lowered):
                return True
            return False
        except Exception:
            return False

    def _infer_with_retries(self, prompt: str, image_path: Path, image_obj=None, max_attempts: int = 3, initial_backoff: float = 1.0) -> str:
        """Call the model up to max_attempts, skipping HTML error pages."""
        backoff = initial_backoff
        last_text = ""
        for attempt in range(1, max_attempts + 1):
            try:
                if self.client is not None:
                    text = self.client.infer(prompt, Path(image_path))
                elif self.model is not None:
                    # Use existing image object if available
                    response = self.model.generate_content([prompt, image_obj]) if image_obj is not None else self.model.generate_content([prompt, Image.open(image_path)])
                    text = getattr(response, 'text', str(response))
                else:
                    raise RuntimeError("No model/client initialized for inference")
            except Exception as e:
                text = f"Error occurred during inference: {str(e)}"

            if not self._is_html_error_text(text):
                return text

            print(f"  Attempt {attempt}/{max_attempts} returned HTML error; retrying after {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            last_text = text
        return last_text

    def _build_client(self, provider: str, model_name: str, temperature: float):
        provider_key = provider.lower()
        if provider_key in ('route', 'routellm'):
            return _RouteLLMVisionClient(model_name or 'gemini-2.5-pro', temperature, max_tokens=ROUTELLM_MAX_TOKENS)
        if provider_key in ('gemini', 'google', 'gemini-2.5', 'gemini-2.0'):
            #return _GeminiVisionClient(model_name or 'gemini-2.5-pro-preview-06-05', temperature)
            return _GeminiVisionClient(model_name or 'gemini-3-pro-preview', temperature)
        if provider_key in ('claude', 'anthropic', 'claude-sonnet-4'):
            return _AnthropicVisionClient(model_name or 'claude-sonnet-4-20250514', temperature)
        if provider_key in ('gpt', 'gpt-4o', 'openai'):
            return _OpenAIVisionClient(model_name or 'gpt-4o-2024-05-13', temperature)
        if provider_key in ('openrouter', 'or', 'open-router'):
            return _OpenRouterVisionClient(model_name or 'openai/gpt-4o', temperature)
        # Default fallback
        return _RouteLLMVisionClient('gemini-2.5-pro', temperature)
    
    def _provider_tag(self) -> str:
        key = (self.model_provider or 'gemini').lower()
        if key in ('route', 'routellm'):
            # Prefer concrete model id as tag for routed calls
            tag = (self.model_name or 'route').strip()
            # Sanitize to be filesystem-friendly
            return re.sub(r"[^A-Za-z0-9._-]", "_", tag).lower()
        if key in ('claude', 'anthropic', 'claude-sonnet-4'):
            return 'claude'
        if key in ('gpt', 'gpt-4o', 'openai'):
            return 'gpt'
        if key in ('gemini', 'google', 'gemini-2.5', 'gemini-2.0'):
            # Use the model name when available to match saved files like q*_gemini-2.5-pro.txt
            tag = (self.model_name or 'gemini').strip()
            return re.sub(r"[^A-Za-z0-9._-]", "_", tag).lower()
        if key in ('openrouter', 'or', 'open-router'):
            tag = (self.model_name or 'openrouter').strip()
            return re.sub(r"[^A-Za-z0-9._-]", "_", tag).lower()
        return 'gemini'
        
    def load_question_data(self, question_dir: str) -> Dict:
        """Load all files for a single question."""
        base_path = Path(self.dataset_path) / question_dir
        question_num = int(question_dir[1:])  # Extract number from q1, q2, etc.
        
        # Load question text
        with open(base_path / f"{question_dir}_question.txt", 'r') as f:
            question = f.read().strip()
            
        # Load correct answer (symbolic equation)
        with open(base_path / f"{question_dir}_ta.txt", 'r') as f:
            correct_answer = f.read().strip()
            
        # Load image
        image_path = base_path / f"{question_dir}_image.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Could not find image for question {question_dir}")
        
        image = Image.open(image_path)
        
        return {
            'question': question,
            'correct_answer': correct_answer,
            'image': image,
            'image_path': str(image_path)
        }
    
    def extract_equation_from_response(self, response_text: str) -> Tuple[str, Optional[str]]:
        """
        Extract the symbolic equation and thinking process from Gemini's response.
        Returns: (equation, thinking_process)
        """
        response_text = response_text.strip()
        
        # Extract content between <think> tags
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, response_text, re.DOTALL | re.IGNORECASE)
        thinking = think_match.group(1).strip() if think_match else None
        
        # Extract content between <answer> tags
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if answer_match:
            model_answer = answer_match.group(1).strip()
            
            # Clean up the answer
            model_answer = model_answer.replace("**", "^")
            
            # If the answer doesn't contain an equals sign, it might be just the RHS
            # Try to find the full equation format from the thinking section
            if "=" not in model_answer and thinking:
                # Look for the equation format in thinking
                eq_patterns = [
                    r"H\(s\)\s*=\s*" + re.escape(model_answer),
                    r"Vn\d+\(s\)\s*=\s*" + re.escape(model_answer),
                    r"Z\(s\)\s*=\s*" + re.escape(model_answer),
                    r"I\(s\)\s*=\s*" + re.escape(model_answer),
                ]
                
                for pattern in eq_patterns:
                    if re.search(pattern, thinking, re.IGNORECASE):
                        # Extract the proper format
                        format_match = re.search(r"((?:H|Vn\d+|Z|I)\(s\))", thinking, re.IGNORECASE)
                        if format_match:
                            model_answer = f"{format_match.group(1)} = {model_answer}"
                            break
            
            return model_answer, thinking
        
        # Fallback if no answer tags found
        return self.extract_equation_fallback(response_text), thinking
    
    def extract_equation_fallback(self, response_text: str) -> str:
        """
        Fallback extraction when answer tags are not properly used.
        Tries multiple patterns to find the equation.
        """
        response_text = response_text.strip()
        
        # First try to find complete equations with left and right sides
        complete_eq_patterns = [
            r"(H\(s\)\s*=\s*.+?)(?:\n|$)",
            r"(Vn\d+\(s\)\s*=\s*.+?)(?:\n|$)",
            r"(Z\(s\)\s*=\s*.+?)(?:\n|$)",
            r"(I\(s\)\s*=\s*.+?)(?:\n|$)",
        ]
        
        for pattern in complete_eq_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                equation = match.group(1).strip()
                equation = equation.replace("**", "^")
                return equation
        
        # Then try generic answer patterns
        answer_patterns = [
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"Answer:\s*(.+?)(?:\n|$)",
            r"Therefore,?\s*(.+?)(?:\n|$)",
            r"The (?:transfer function|nodal equation|equation) is:?\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if matches:
                equation = matches[-1].strip()  # Take the last match
                equation = equation.replace("**", "^")
                return equation
        
        # Try to find any equation with '=' sign
        lines = response_text.split('\n')
        for line in reversed(lines):  # Start from the end
            if '=' in line and (line.count('(') == line.count(')')):
                # Check if it looks like an equation (not just text with =)
                if any(char in line for char in ['s', 'R', 'C', 'L', '+', '-', '*', '/']):
                    equation = line.strip()
                    equation = equation.replace("**", "^")
                    return equation
        
        # Last resort: return the last non-empty line that looks mathematical
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        for line in reversed(lines):
            if any(char in line for char in ['=', 's', 'R', 'C', 'L', '(', ')']):
                equation = line.strip()
                equation = equation.replace("**", "^")
                return equation
        
        return response_text.strip()
    
    def clean_equation_for_comparison(self, equation: str) -> str:
        """
        Clean and normalize equation for comparison.
        Removes the LHS (like H(s) = ) and returns just the expression.
        """
        # Start with a trimmed copy
        eq_text = (equation or "").strip()

        # Normalize stray unicode math symbols and brackets early
        eq_text = (
            eq_text
            .replace("×", "*")
            .replace("·", "*")
            .replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
        )
        # Convert square brackets to parentheses (common in informal math)
        eq_text = eq_text.replace("[", "(").replace("]", ")")

        # If multi-line, pick the last line that looks like math; avoid commentary
        lines = [ln.strip() for ln in eq_text.splitlines() if ln.strip()]
        if len(lines) > 1:
            math_line_candidates = [
                ln for ln in lines
                if ("=" in ln) or re.search(r"[+\-*/()]+", ln) or re.search(r"\b[sRCLGIVAd]\b", ln, re.IGNORECASE)
            ]
            if math_line_candidates:
                eq_text = math_line_candidates[-1]
            else:
                eq_text = lines[0]

        # Drop trailing commentary markers like "Where ...", "Thus ...", etc.
        commentary_split = re.split(r"\b(where|which|with|subject to|assuming|let\s|given that|thus|therefore)\b",
                                    eq_text, flags=re.IGNORECASE)
        if commentary_split:
            eq_text = commentary_split[0].strip()

        # Collapse any duplicate equals or chain equalities by taking RHS of the last '='
        if "=" in eq_text:
            eq_text = eq_text.rsplit("=", 1)[-1].strip()

        # Remove common LHS patterns if any still remain (defensive)
        lhs_patterns = [
            r"H\(s\)\s*=\s*",
            r"Vn\d+\(s\)\s*=\s*",
            r"Z\(s\)\s*=\s*",
            r"I\(s\)\s*=\s*",
            r"V\(s\)\s*=\s*",
            r"[A-Za-z_]\w*\(s\)\s*=\s*",  # Generic single function on LHS like iv1(s) =
            r"I_R\d+\(s\)/V\d+\(s\)\s*=\s*",  # Transfer function formats like I_R3(s)/V1(s) =
            r"V_R\d+\(s\)/V\d+\(s\)\s*=\s*",  # Voltage transfer functions
            r"[A-Za-z_]\w*\(s\)/[A-Za-z_]\w*\(s\)\s*=\s*",  # General transfer function format
        ]
        for pattern in lhs_patterns:
            eq_text = re.sub(pattern, "", eq_text, flags=re.IGNORECASE)

        # Remove trailing punctuation that's not part of math
        eq_text = re.sub(r"[;,:\.]\s*$", "", eq_text)

        # Canonical caret usage (downstream will convert to **)
        eq_text = eq_text.replace("**", "^").strip()

        return eq_text

    def _normalize_latex_to_ascii(self, equation: str) -> str:
        """Best-effort conversion of common LaTeX to SymPy-friendly ASCII.
        This avoids requiring sympy latex parser and handles the most frequent patterns.
        """
        if not equation:
            return equation

        eq = equation
        # Strip LaTeX math delimiters (inline and block) anywhere in the string
        eq = eq.strip()
        eq = re.sub(r"\\\(|\\\)", "", eq)  # remove \( \)
        eq = re.sub(r"\\\[|\\\]", "", eq)  # remove \[ \]
        eq = eq.replace("$", "")  # remove $ or $$
        # Remove LaTeX line breaks and alignment markers
        eq = eq.replace("\\\\", " ")
        eq = eq.replace("&", " ")
        # Remove inline LaTeX comments
        eq = re.sub(r"%.*", "", eq)

        eq = eq.replace("\\left", "").replace("\\right", "")
        eq = eq.replace("\\cdot", "*").replace("\\times", "*")
        # Normalize common unicode math symbols as well
        eq = eq.replace("×", "*").replace("·", "*")
        eq = eq.replace("−", "-").replace("–", "-").replace("—", "-")
        eq = eq.replace("\\,", "").replace("\\!", "")

        # Replace \frac{a}{b} with (a)/(b) iteratively (handles nested reasonably)
        def replace_frac(text: str) -> str:
            pattern = re.compile(r"\\frac\{([^{}]+|\{[^{}]*\})+\}\{([^{}]+|\{[^{}]*\})+\}")
            # Simpler non-greedy balanced approach using a stack-based scan
            i = 0
            out = []
            while i < len(text):
                if text.startswith("\\frac{", i):
                    # Find numerator
                    i0 = i + len("\\frac{")
                    depth = 1
                    j = i0
                    while j < len(text) and depth > 0:
                        if text[j] == '{':
                            depth += 1
                        elif text[j] == '}':
                            depth -= 1
                        j += 1
                    num = text[i0:j-1]
                    # Expect '/{'
                    if j < len(text) and text[j] == '{':
                        depth = 1
                        k = j + 1
                        while k < len(text) and depth > 0:
                            if text[k] == '{':
                                depth += 1
                            elif text[k] == '}':
                                depth -= 1
                            k += 1
                        den = text[j+1:k-1]
                        out.append('(')
                        out.append(num)
                        out.append(')')
                        out.append('/')
                        out.append('(')
                        out.append(den)
                        out.append(')')
                        i = k
                        continue
                out.append(text[i])
                i += 1
            return ''.join(out)

        prev = None
        while prev != eq:
            prev = eq
            eq = replace_frac(eq)

        # Remove any remaining simple LaTeX commands (after handling \frac)
        eq = re.sub(r"\\[a-zA-Z]+", "", eq)

        # Remove LaTeX braces when safe
        eq = eq.replace('{', '(').replace('}', ')')

        # Convert subscripts: X_12 -> X12; also forms like i_{V1} -> IV1, v_{R3} -> VR3
        eq = re.sub(r"([A-Za-z])_\{?([A-Za-z]+\d+)\}?", r"\1\2", eq)
        # Generic join for alpha-numeric subscripts e.g., R_4 -> R4, L_{2} -> L2, G_{m} -> Gm
        eq = re.sub(r"([A-Za-z]+)_\{?([A-Za-z0-9]+)\}?", r"\1\2", eq)
        eq = re.sub(r"\bi_\{?V(\d+)\}?", r"IV\1", eq, flags=re.IGNORECASE)
        eq = re.sub(r"\bv_\{?V(\d+)\}?", r"VV\1", eq, flags=re.IGNORECASE)
        eq = re.sub(r"\bv_\{?R(\d+)\}?", r"VR\1", eq, flags=re.IGNORECASE)
        eq = re.sub(r"\bi_\{?R(\d+)\}?", r"IR\1", eq, flags=re.IGNORECASE)

        # Normalize V_1(s) -> V1, IV1(s) -> IV1, V1(s) -> V1
        eq = re.sub(r"\b([A-Za-z]+\d+)\(s\)", r"\1", eq)

        # Replace caret to python exponent later, but ensure we don't lose info here
        return eq

    def preprocess_for_sympy(self, equation: str) -> str:
        """Pipeline to prepare equation string for SymPy parsing."""
        eq = equation or ""
        eq = eq.strip()
        # Common LaTeX to ASCII
        eq = self._normalize_latex_to_ascii(eq)
        # Convert square brackets and braces to parentheses when safe
        eq = eq.replace("[", "(").replace("]", ")")
        # Normalize whitespace
        eq = re.sub(r"\s+", " ", eq)
        # Convert caret to python exponent
        eq = eq.replace('^', '**')
        # Common implicit multiplication hints: sC1 -> s*C1, R1C1 -> R1*C1 (handled by implicit mul)
        return eq
    
    @timeout(60)  # 60 second timeout for simplification
    def safe_simplify(self, expr):
        """Safely simplify an expression with timeout."""
        return sp.simplify(expr)
    
    @timeout(60)  # 60 second timeout for expansion
    def safe_expand(self, expr):
        """Safely expand an expression with timeout."""
        return sp.expand(expr)
    
    @timeout(STRATEGY_TIMEOUT_SECONDS)  # per-strategy timeout
    def safe_strategy(self, strategy_func):
        """Safely execute a comparison strategy with timeout."""
        return strategy_func()
    
    def compare_symbolic_equations(self, model_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """Compare two symbolic equations using SymPy with timeout protection."""
        try:
            # Clean both equations for comparison (remove LHS)
            model_cleaned = self.clean_equation_for_comparison(model_answer)
            correct_cleaned = self.clean_equation_for_comparison(correct_answer)
            
            print(f"Comparing (after cleaning):")
            print(f"  Model:   {model_cleaned}")
            print(f"  Correct: {correct_cleaned}")
            
            # Replace ^ with ** for SymPy parsing
            model_cleaned = model_cleaned.replace("^", "**")
            correct_cleaned = correct_cleaned.replace("^", "**")
            
            # Preprocess for parsing
            model_ready = self.preprocess_for_sympy(model_cleaned)
            correct_ready = self.preprocess_for_sympy(correct_cleaned)

            # Build a local symbol table so tokens like R1, C2, V1 are parsed as single symbols (not R*1)
            def collect_symbol_names(*texts: List[str]) -> List[str]:
                token_pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
                reserved = {"s", "I", "E", "pi", "oo", "zoo", "nan", "O", "ComplexInfinity"}
                names: set = set()
                for t in texts:
                    if not t:
                        continue
                    for tok in token_pattern.findall(t):
                        # Skip obvious function names or reserved constants
                        if tok in reserved:
                            continue
                        # Heuristics: keep typical circuit tokens (R1, C2, L3, V1, IV1, VR3, Ad, Gm, etc.)
                        # Allow multi-letter names with digits; avoid single-letter variables commonly reserved
                        names.add(tok)
                return sorted(names, key=lambda x: (len(x), x))

            sym_names = collect_symbol_names(model_ready, correct_ready)
            local_dict = {name: sp.Symbol(name) for name in sym_names}

            # Parse with SymPy using implicit multiplication and XOR conversion
            transformations = standard_transformations + (
                implicit_multiplication_application,
                convert_xor,
            )
            try:
                model_expr = parse_expr(model_ready, transformations=transformations, evaluate=False, local_dict=local_dict)
                correct_expr = parse_expr(correct_ready, transformations=transformations, evaluate=False, local_dict=local_dict)
            except Exception as e:
                # Fallback to sympify if parse_expr fails
                try:
                    model_expr = sp.sympify(model_ready, locals=local_dict, evaluate=False)
                    correct_expr = sp.sympify(correct_ready, locals=local_dict, evaluate=False)
                except Exception as ee:
                    return False, f"Parsing error: {str(e)} | Fallback: {str(ee)}"
            
            # Try multiple comparison strategies with timeouts
            strategies = [
                ("Direct equality", lambda: model_expr.equals(correct_expr)),
                ("Simple difference", lambda: sp.simplify(model_expr - correct_expr).equals(0)),
                ("Basic difference check", lambda: (model_expr - correct_expr).equals(0)),
                ("Expand and compare", lambda: sp.expand(model_expr - correct_expr).equals(0)),
                ("Factor and compare", lambda: sp.factor(model_expr - correct_expr).equals(0)),
                # Sign-invariant checks
                ("Direct equality up to sign", lambda: model_expr.equals(-correct_expr)),
                ("Sum equals zero (up to sign)", lambda: sp.simplify(model_expr + correct_expr).equals(0)),
                ("Expand and compare (sum, up to sign)", lambda: sp.expand(model_expr + correct_expr).equals(0)),
                ("Factor and compare (sum, up to sign)", lambda: sp.factor(model_expr + correct_expr).equals(0)),
                ("Ratio equals -1 (up to sign)", lambda: sp.simplify(sp.together(model_expr / correct_expr) + 1).equals(0)),
            ]
            
            compare_start_time = time.time()
            for strategy_name, strategy_func in strategies:
                try:
                    print(f"  Trying strategy: {strategy_name}...")
                    start_time = time.time()
                    is_equal = self.safe_strategy(strategy_func)
                    elapsed = time.time() - start_time
                    print(f"    Strategy completed in {elapsed:.2f}s")
                    if is_equal:
                        return True, f"Equal using {strategy_name}"
                except TimeoutError:
                    print(f"    Strategy {strategy_name} timed out after {STRATEGY_TIMEOUT_SECONDS}s, trying next...")
                    continue
                except Exception as e:
                    print(f"    Strategy {strategy_name} failed: {str(e)}, trying next...")
                    continue
                # Note: global wall-clock timeout is enforced by a separate process wrapper
            
            # If all strategies fail or timeout, fall back to numerical comparison immediately
            print("  All symbolic strategies failed or timed out, trying numerical evaluation...")
            return self.numerical_comparison_fallback(model_expr, correct_expr)
                
        except Exception as e:
            return False, f"Comparison error: {str(e)}"

    @staticmethod
    def _compare_worker(model_answer: str, correct_answer: str, result_queue: Queue):
        """Worker to run comparison in a separate process to allow hard timeout."""
        try:
            bench = SymbolicEquationBenchmark(init_model=False)
            result = bench.compare_symbolic_equations(model_answer, correct_answer)
            result_queue.put(result)
        except Exception as e:
            result_queue.put((None, f"Worker error: {str(e)}"))

    def compare_with_global_timeout(self, model_answer: str, correct_answer: str, timeout_seconds: int = GLOBAL_COMPARE_TIMEOUT_SECONDS):
        """Run comparison in a separate process and enforce a hard wall-clock timeout.

        Returns (is_correct, details). If timed out, is_correct is None and the sample should be excluded from accuracy.
        """
        q: Queue = Queue()
        p = Process(target=SymbolicEquationBenchmark._compare_worker, args=(model_answer, correct_answer, q))
        p.start()
        p.join(timeout_seconds)
        if p.is_alive():
            print(f"  Comparison exceeded {timeout_seconds}s. Terminating worker and skipping this sample.")
            p.terminate()
            p.join()
            return None, f"Comparison timed out after {timeout_seconds}s; skipped"
        if q.empty():
            return None, "Comparison failed: no result returned"
        return q.get()
    
    def numerical_comparison_fallback(self, model_expr, correct_expr) -> Tuple[bool, str]:
        """Fallback to numerical evaluation for complex expressions."""
        try:
            print("    Starting numerical comparison...")
            # Extract all symbols from both expressions
            model_symbols = model_expr.free_symbols
            correct_symbols = correct_expr.free_symbols
            all_symbols = model_symbols.union(correct_symbols)
            
            # Also include applied undefined functions like V1(s), Iin(s), etc.
            model_applied_funcs = model_expr.atoms(AppliedUndef)
            correct_applied_funcs = correct_expr.atoms(AppliedUndef)
            all_applied_funcs = model_applied_funcs.union(correct_applied_funcs)
            
            # Remove 's' (Laplace variable) from symbols to evaluate
            symbols_to_evaluate = {sym for sym in all_symbols if str(sym) != 's'}
            
            if not symbols_to_evaluate:
                return False, "No variables found for numerical evaluation"
            
            print(f"    Found {len(symbols_to_evaluate)} symbols to evaluate: {[str(s) for s in symbols_to_evaluate]}")
            
            # Generate random test values for the symbols
            import random
            random.seed(42)  # For reproducible results
            
            # Robust numerical settings
            num_tests = 36  # diversified grid
            tolerance_abs = 1e-5
            tolerance_rel = 5e-3  # 0.5%
            successful_tests = 0
            sign_votes_pos = 0
            sign_votes_neg = 0
            
            for test_num in range(num_tests):
                # Generate random values for symbols (avoiding zeros for denominators)
                test_values = {}
                for sym in symbols_to_evaluate:
                    sym_str = str(sym)
                    if 'R' in sym_str:  # Resistors
                        test_values[sym] = random.uniform(1.0, 1000.0)
                    elif 'C' in sym_str:  # Capacitors
                        test_values[sym] = random.uniform(1e-9, 1e-6)
                    elif 'L' in sym_str:  # Inductors
                        test_values[sym] = random.uniform(1e-6, 1e-3)
                    elif 'G' in sym_str:  # Conductances
                        test_values[sym] = random.uniform(1e-3, 1.0)
                    elif 'Ad' in sym_str:  # Amplifier gains
                        test_values[sym] = random.uniform(1.0, 100.0)
                    elif 'Cint' in sym_str:  # Internal capacitances
                        test_values[sym] = random.uniform(1e-12, 1e-9)
                    elif 'Rint' in sym_str:  # Internal resistances
                        test_values[sym] = random.uniform(1.0, 100.0)
                    else:  # Other variables
                        test_values[sym] = random.uniform(0.1, 10.0)
                
                # Assign numeric values to applied undefined functions like V1(s)
                for f in all_applied_funcs:
                    test_values[f] = random.uniform(0.1, 10.0)

                # Cycle through diversified s points to avoid singularities and capture sign stability
                s_grid = [
                    0.1j, 0.2j, 0.5j, 1j, 2j, 5j, 10j,
                    0.1 + 0.1j, 0.2 + 0.2j, 0.5 + 0.5j, 1 + 1j, 2 + 2j, 5 + 5j,
                    -0.1 + 0.1j, -0.5 + 0.5j, -1 + 1j, -2 + 2j, -5 + 5j,
                    0.1 - 0.1j, 0.5 - 0.5j, 1 - 1j, 2 - 2j, 5 - 5j,
                    0.05j, 0.3j, 3j, 7j, 15j,
                    0.1 + 0.05j, 0.5 + 0.2j, 1 + 0.3j, 2 + 0.1j, -1 + 0.2j, -2 + 0.3j
                ]
                test_values[sp.Symbol('s')] = s_grid[test_num % len(s_grid)]
                
                try:
                    # Evaluate both expressions with timeout protection
                    @timeout(10)  # 10 second timeout per evaluation
                    def evaluate_expressions():
                        model_val = complex(model_expr.evalf(subs=test_values))
                        correct_val = complex(correct_expr.evalf(subs=test_values))
                        return model_val, correct_val
                    
                    model_val, correct_val = evaluate_expressions()
                    # Sign-invariant closeness check
                    err_abs_1 = abs(model_val - correct_val)
                    err_abs_2 = abs(model_val + correct_val)
                    err_abs = min(err_abs_1, err_abs_2)
                    denom = max(abs(correct_val), abs(model_val), 1e-10)
                    rel_error = err_abs / denom

                    if (err_abs <= tolerance_abs) or (rel_error <= tolerance_rel):
                        successful_tests += 1
                        print(f"    Test {test_num + 1}: PASS abs={err_abs:.2e} rel={rel_error:.2e}")
                        # vote sign using a stable rule based on which diff is smaller
                        if err_abs_1 <= err_abs_2:
                            sign_votes_pos += 1  # model ~ +correct
                        else:
                            sign_votes_neg += 1  # model ~ -correct
                    else:
                        print(f"    Test {test_num + 1}: FAIL abs={err_abs:.2e} rel={rel_error:.2e}")
                        
                except TimeoutError:
                    print(f"    Numerical evaluation timed out for test {test_num + 1}")
                    continue
                except Exception as e:
                    print(f"    Numerical evaluation failed for test {test_num + 1}: {str(e)}")
                    continue
            
            if successful_tests == 0:
                return False, "Numerical comparison inconclusive: no successful evaluations (possibly due to undefined functions like V1(s))"

            # Require a minimum pass rate to be more robust
            min_pass_rate = 0.6
            pass_rate = successful_tests / num_tests
            if pass_rate < min_pass_rate:
                return False, f"Numerical mismatch (up to sign). Pass rate {pass_rate:.0%} < {int(min_pass_rate*100)}%"

            # Report sign consistency if informative
            dominant_sign = "+" if sign_votes_pos >= sign_votes_neg else "-"
            return True, (
                f"Numerically equal (up to sign) with abs<= {tolerance_abs} or rel<= {tolerance_rel:.2%}; "
                f"passes {successful_tests}/{num_tests} ({pass_rate:.0%}), dominant sign {dominant_sign}"
            )
            
        except Exception as e:
            return False, f"Numerical comparison error: {str(e)}"
    
    def save_response_to_file(self, question_dir: str, response_text: str, model_tag: Optional[str] = None) -> None:
        """Save the model's response to a text file in the question folder."""
        try:
            tag = (model_tag or self._provider_tag()).lower()
            output_path = Path(self.dataset_path) / question_dir / f"{question_dir}_{tag}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            print(f"  Saved response to: {output_path}")
        except Exception as e:
            print(f"  Warning: Could not save response to file: {str(e)}")
    
    def load_response_from_file(self, question_dir: str, model_tag: Optional[str] = None) -> str:
        """Load the model's response from a text file in the question folder."""
        try:
            tag = (model_tag or self._provider_tag()).lower()
            response_path = Path(self.dataset_path) / question_dir / f"{question_dir}_{tag}.txt"
            if not response_path.exists():
                raise FileNotFoundError(f"No saved response found for {question_dir} ({tag})")
            
            with open(response_path, 'r', encoding='utf-8') as f:
                response_text = f.read()
            return response_text
        except Exception as e:
            print(f"  Error loading response for {question_dir}: {str(e)}")
            return ""
    
    def evaluate_question(self, question_data: Dict, question_dir: str) -> Dict:
        """Evaluate a single question using Gemini."""
        # Enhanced prompt with think/answer format
        prompt = f"""You are an expert electrical engineer specializing in circuit analysis. Analyze the circuit diagram and solve for the requested symbolic expression.

        **Task:** {question_data['question']}

        **Critical Instructions:**
        1. Use EXACT component labels as shown in the circuit (e.g., R1, R2, C1, C2, L1, not generic R, C, L)
        2. For Laplace domain, use lowercase 's' as the complex frequency variable
        3. Use standard impedances: R for resistors, 1/(s*C) for capacitors, s*L for inductors
        4. For op-amps: Apply virtual short (V+ = V-) if in negative feedback, use Ad for gain if specified
        5. Simplify the final expression to match standard form

        **Response Format:**
        You MUST structure your response exactly as follows:

        <think>
        [Show all your reasoning and intermediate steps here]
        </think>

        <answer>
        [Only the final symbolic equation here in the form: H(s) = expression, or Vn1(s) = expression, etc.]
        </answer>

        Use * for multiplication, / for division, and parentheses for grouping. Keep the answer in simplest form."""

        try:
            # Use selected model provider client
            if self.client is not None:
                response_text = self.client.infer(prompt, Path(question_data.get('image_path')))
            elif self.model is not None:  # fallback to gemini model
                response = self.model.generate_content([prompt, question_data['image']])
                response_text = response.text
            else:
                raise RuntimeError("No model/client initialized for inference")
            
            # Save the full response to a file in the question folder with provider tag
            self.save_response_to_file(question_dir, response_text, model_tag=self._provider_tag())
            
            print("\n" + "="*60)
            print("FULL RESPONSE:")
            print("-"*60)
            print(response_text)
            print("="*60)
            
            # Extract equation and thinking process
            model_answer, thinking = self.extract_equation_from_response(response_text)
            
            # Display extracted components
            if thinking:
                print("\nTHINKING PROCESS (first 500 chars):")
                print("-"*60)
                print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
                print("-"*60)
            
            print(f"\nEXTRACTED EQUATION: {model_answer}")
            print(f"EXPECTED ANSWER:    {question_data['correct_answer']}")
            
            # Compare with correct answer using SymPy with a hard global timeout
            is_correct, comparison_details = self.compare_with_global_timeout(
                model_answer, question_data['correct_answer']
            )
            
            print(f"\nIS CORRECT: {is_correct}")
            print(f"DETAILS: {comparison_details}")
            
            return {
                'model_answer': model_answer,
                'correct_answer': question_data['correct_answer'],
                'is_correct': is_correct,
                'comparison_details': comparison_details,
                'thinking_process': thinking,
                'full_response': response_text
            }
            
        except Exception as e:
            print(f"Error evaluating question: {str(e)}")
            # Save error information to file as well
            error_text = f"Error occurred: {str(e)}\n\nNo response generated."
            self.save_response_to_file(question_dir, error_text)
            
            return {
                'error': str(e),
                'model_answer': None,
                'correct_answer': question_data['correct_answer'],
                'is_correct': False,
                'thinking_process': None
            }
    
    def run_inference_only(self, max_questions: int = 600) -> pd.DataFrame:
        """Run inference only and save responses to files."""
        # Get all question directories
        question_dirs = [d for d in os.listdir(self.dataset_path) 
                        if os.path.isdir(os.path.join(self.dataset_path, d)) 
                        and d.startswith('q')]
        
        # Sort by question number
        question_dirs.sort(key=lambda x: int(x[1:]))
        
        # Limit to first max_questions
        question_dirs = question_dirs[:max_questions]
        print(f"Running inference on {len(question_dirs)} questions: {question_dirs[:10]}{'...' if len(question_dirs) > 10 else ''}")
        
        start_time = time.time()
        inference_results = []
        
        for i, q_dir in enumerate(question_dirs):
            print(f"\n{'='*70}")
            print(f"INFERENCE {q_dir} ({i+1}/{len(question_dirs)})")
            print(f"Elapsed time: {time.time() - start_time:.1f} seconds")
            print(f"{'='*70}")
            
            try:
                # Check if response already exists for current provider; do not skip, overwrite
                response_path = Path(self.dataset_path) / q_dir / f"{q_dir}_{self._provider_tag()}.txt"
                if response_path.exists():
                    try:
                        existing_text = response_path.read_text(encoding='utf-8', errors='ignore')
                    except Exception:
                        existing_text = ""
                    if not self._is_html_error_text(existing_text):
                        print(f"  Response already exists for {q_dir}, overwriting...")
                    else:
                        print(f"  Existing response for {q_dir} looks like an HTML error. Re-running...")
                
                # Load question data
                question_data = self.load_question_data(q_dir)
                print(f"Question: {question_data['question']}")
                
                # Generate response using selected provider
                prompt = f"""You are an expert electrical engineer specializing in circuit analysis. Analyze the circuit diagram and solve for the requested symbolic expression.

                **Task:** {question_data['question']}

                **Critical Instructions:**
                1. Use EXACT component labels as shown in the circuit (e.g., R1, R2, C1, C2, L1, not generic R, C, L)
                2. For Laplace domain, use lowercase 's' as the complex frequency variable
                3. Use standard impedances: R for resistors, 1/(s*C) for capacitors, s*L for inductors
                4. For op-amps: Apply virtual short (V+ = V-) if in negative feedback, use Ad for gain if specified
                5. Simplify the final expression to match standard form

                **Response Format:**
                You MUST structure your response exactly as follows:

                <think>
                [Show all your reasoning and intermediate steps here]
                - Identify all components and their labels from the circuit
                - Define impedances for each component
                - Apply appropriate circuit analysis method (nodal, mesh, etc.)
                - Write out all equations step by step
                - Show algebraic manipulation
                - Simplify to get the final form
                </think>

                <answer>
                [Only the final symbolic equation here in the form: H(s) = expression, or Vn1(s) = expression, etc.]
                </answer>

                Use * for multiplication, / for division, and parentheses for grouping. Keep the answer in simplest form."""

                response_text = self._infer_with_retries(prompt, Path(question_data.get('image_path')), image_obj=question_data.get('image'))

                if self._is_html_error_text(response_text):
                    print(f"✗ Still receiving HTML error after retries for {q_dir}; not writing file")
                    inference_results.append({
                        'question_number': q_dir,
                        'status': 'html_error',
                        'processing_time': time.time() - start_time
                    })
                else:
                    # Save the response to file with provider tag (overwrite if exists)
                    self.save_response_to_file(q_dir, response_text, model_tag=self._provider_tag())
                    print(f"✓ Generated and saved response for {q_dir} (overwritten if existed)")
                    inference_results.append({
                        'question_number': q_dir,
                        'status': 'completed',
                        'processing_time': time.time() - start_time,
                        'response_length': len(response_text)
                    })
                
            except Exception as e:
                print(f"✗ ERROR generating response for {q_dir}: {str(e)}")
                # Save error information to file with provider tag
                error_text = f"Error occurred during inference: {str(e)}\n\nNo response generated."
                self.save_response_to_file(q_dir, error_text, model_tag=self._provider_tag())
                
                inference_results.append({
                    'question_number': q_dir,
                    'status': 'error',
                    'error': str(e),
                    'processing_time': time.time() - start_time
                })
            
            # Save intermediate results every 10 questions
            if (i + 1) % 10 == 0:
                temp_df = pd.DataFrame(inference_results)
                temp_df.to_csv(f'results/temp_inference_{i+1}.csv', index=False)
                print(f"Intermediate inference results saved to temp_inference_{i+1}.csv")
        
        # Convert results to DataFrame
        df = pd.DataFrame(inference_results)
        
        # Calculate metrics
        total_questions = len(df)
        completed = (df['status'] == 'completed').sum() if 'status' in df.columns else 0
        already_exists = (df['status'] == 'already_exists').sum() if 'status' in df.columns else 0
        errors = (df['status'] == 'error').sum() if 'status' in df.columns else 0
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("INFERENCE RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Total Questions: {total_questions}")
        print(f"Completed: {completed}")
        print(f"Already Existed: {already_exists}")
        print(f"Errors: {errors}")
        print(f"Total Time: {total_time:.1f} seconds")
        print(f"Average Time per Question: {total_time/total_questions:.1f} seconds")
        
        return df
    
    def run_retry_html_errors_only(self, max_questions: int = 600) -> pd.DataFrame:
        """Re-run inference only for questions whose saved response is an HTML error."""
        # Get all question directories
        question_dirs = [d for d in os.listdir(self.dataset_path) 
                        if os.path.isdir(os.path.join(self.dataset_path, d)) 
                        and d.startswith('q')]
        
        # Sort by question number
        question_dirs.sort(key=lambda x: int(x[1:]))
        
        # Limit
        question_dirs = question_dirs[:max_questions]
        print(f"Retrying HTML-error responses among {len(question_dirs)} questions")
        
        start_time = time.time()
        retry_results = []
        
        for i, q_dir in enumerate(question_dirs):
            try:
                response_path = Path(self.dataset_path) / q_dir / f"{q_dir}_{self._provider_tag()}.txt"
                if not response_path.exists():
                    continue
                try:
                    existing_text = response_path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    existing_text = ""
                if not self._is_html_error_text(existing_text):
                    continue
                
                print(f"\n{'='*70}")
                print(f"RETRY {q_dir} ({i+1}/{len(question_dirs)})")
                print(f"{'='*70}")
                
                question_data = self.load_question_data(q_dir)
                prompt = f"""You are an expert electrical engineer specializing in circuit analysis. Analyze the circuit diagram and solve for the requested symbolic expression.

        **Task:** {question_data['question']}

        **Critical Instructions:**
        1. Use EXACT component labels as shown in the circuit (e.g., R1, R2, C1, C2, L1, not generic R, C, L)
        2. For Laplace domain, use lowercase 's' as the complex frequency variable
        3. Use standard impedances: R for resistors, 1/(s*C) for capacitors, s*L for inductors
        4. For op-amps: Apply virtual short (V+ = V-) if in negative feedback, use Ad for gain if specified
        5. Simplify the final expression to match standard form

        **Response Format:**
        You MUST structure your response exactly as follows:

        <think>
        [Show all your reasoning and intermediate steps here]
        </think>

        <answer>
        [Only the final symbolic equation here in the form: H(s) = expression, or Vn1(s) = expression, etc.]
        </answer>

        Use * for multiplication, / for division, and parentheses for grouping. Keep the answer in simplest form."""
                
                response_text = self._infer_with_retries(prompt, Path(question_data.get('image_path')), image_obj=question_data.get('image'))

                if self._is_html_error_text(response_text):
                    print(f"✗ Still receiving HTML error after retries for {q_dir}; leaving previous file untouched")
                    status = 'retried_html_error'
                else:
                    # Save overwrite with clean content
                    self.save_response_to_file(q_dir, response_text, model_tag=self._provider_tag())
                    status = 'retried_ok'
                retry_results.append({
                    'question_number': q_dir,
                    'status': status,
                    'processing_time': time.time() - start_time,
                    'response_length': (len(response_text) if not self._is_html_error_text(response_text) else 0)
                })
            except Exception as e:
                print(f"✗ ERROR retrying {q_dir}: {str(e)}")
                retry_results.append({
                    'question_number': q_dir,
                    'status': 'error',
                    'error': str(e),
                    'processing_time': time.time() - start_time
                })
        
        df = pd.DataFrame(retry_results)
        return df
    
    def run_evaluation_only(self, max_questions: int = 600) -> pd.DataFrame:
        """Run evaluation only on existing saved responses."""
        # Get all question directories
        question_dirs = [d for d in os.listdir(self.dataset_path) 
                        if os.path.isdir(os.path.join(self.dataset_path, d)) 
                        and d.startswith('q')]
        
        # Sort by question number
        question_dirs.sort(key=lambda x: int(x[1:]))
        
        # Limit to first max_questions
        question_dirs = question_dirs[:max_questions]
        print(f"Evaluating {len(question_dirs)} questions: {question_dirs[:10]}{'...' if len(question_dirs) > 10 else ''}")
        
        start_time = time.time()
        
        for i, q_dir in enumerate(question_dirs):
            print(f"\n{'='*70}")
            print(f"EVALUATION {q_dir} ({i+1}/{len(question_dirs)})")
            print(f"Elapsed time: {time.time() - start_time:.1f} seconds")
            print(f"{'='*70}")
            
            try:
                # Load question data
                question_data = self.load_question_data(q_dir)
                print(f"Question: {question_data['question']}")
                
                # Load existing response for current provider
                response_text = self.load_response_from_file(q_dir, model_tag=self._provider_tag())
                if not response_text:
                    print(f"✗ No saved response found for {q_dir}")
                    self.results.append({
                        'question_number': q_dir,
                        'question_text': question_data['question'],
                        'error': 'No saved response found',
                        'is_correct': False,
                        'processing_time': time.time() - start_time
                    })
                    continue
                
                # Extract equation and thinking process
                model_answer, thinking = self.extract_equation_from_response(response_text)
                
                # Display extracted components
                if thinking:
                    print("\nTHINKING PROCESS (first 200 chars):")
                    print("-"*40)
                    print(thinking[:200] + "..." if len(thinking) > 200 else thinking)
                    print("-"*40)
                
                print(f"\nEXTRACTED EQUATION: {model_answer}")
                print(f"EXPECTED ANSWER:    {question_data['correct_answer']}")
                
                # Compare with correct answer using SymPy with a hard global timeout
                is_correct, comparison_details = self.compare_with_global_timeout(
                    model_answer, question_data['correct_answer']
                )
                
                print(f"\nIS CORRECT: {is_correct}")
                print(f"DETAILS: {comparison_details}")
                
                result = {
                    'question_number': q_dir,
                    'question_text': question_data['question'],
                    'model_answer': model_answer,
                    'correct_answer': question_data['correct_answer'],
                    'is_correct': is_correct,
                    'comparison_details': comparison_details,
                    'thinking_process': thinking,
                    'processing_time': time.time() - start_time
                }
                
                self.results.append(result)
                
                status = "✓ CORRECT" if is_correct is True else ("– SKIPPED" if pd.isna(is_correct) else "✗ INCORRECT")
                print(f"\n{status} - {q_dir}")
                
            except Exception as e:
                print(f"✗ ERROR evaluating {q_dir}: {str(e)}")
                # Try to include question text if readable
                qt = None
                try:
                    qt = (Path(self.dataset_path) / q_dir / f"{q_dir}_question.txt").read_text(encoding='utf-8', errors='ignore').strip()
                except Exception:
                    qt = None
                self.results.append({
                    'question_number': q_dir,
                    'question_text': qt,
                    'error': str(e),
                    'is_correct': False,
                    'processing_time': time.time() - start_time
                })
            
            # Save intermediate results every 10 questions
            if (i + 1) % 10 == 0:
                temp_df = pd.DataFrame(self.results)
                temp_df.to_csv(f'results/temp_evaluation_{i+1}.csv', index=False)
                print(f"Intermediate evaluation results saved to temp_evaluation_{i+1}.csv")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate metrics (exclude skipped/None from accuracy)
        total_questions = len(df)
        evaluated = df['is_correct'].notna().sum() if 'is_correct' in df.columns else 0
        correct_answers = (df['is_correct'] == True).sum() if 'is_correct' in df.columns else 0
        accuracy = (correct_answers / evaluated) if evaluated > 0 else 0
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Total Questions: {total_questions}")
        print(f"Evaluated (excl. skipped): {evaluated}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Total Time: {total_time:.1f} seconds")
        print(f"Average Time per Question: {total_time/total_questions:.1f} seconds")
        
        # Grouped accuracy by question type (transfer vs other)
        if 'question_text' in df.columns:
            try:
                df['is_transfer'] = df['question_text'].str.contains(r'transfer function', case=False, na=False)
                for group_name, group_mask in [("TRANSFER GROUP", df['is_transfer'] == True), ("OTHER GROUP", df['is_transfer'] == False)]:
                    gdf = df[group_mask]
                    g_total = len(gdf)
                    g_evaluated = gdf['is_correct'].notna().sum()
                    g_correct = (gdf['is_correct'] == True).sum()
                    g_accuracy = (g_correct / g_evaluated) if g_evaluated > 0 else 0
                    print(f"\n[{group_name}]")
                    print(f"  Total: {g_total}")
                    print(f"  Evaluated: {g_evaluated}")
                    print(f"  Correct: {g_correct}")
                    print(f"  Accuracy: {g_accuracy:.2%}")
            except Exception as e:
                print(f"Warning: failed to compute grouped accuracy: {str(e)}")
        
        # Show detailed results
        print(f"\nDetailed Results:")
        print("-"*70)
        for _, row in df.iterrows():
            ic = row.get('is_correct', None)
            status = "✓" if ic is True else ("–" if pd.isna(ic) else "✗")
            details_val = row.get('comparison_details', None)
            if details_val is None or (isinstance(details_val, float) and pd.isna(details_val)):
                details_str = 'N/A'
            else:
                details_str = str(details_val)
            if len(details_str) > 50:
                details_str = details_str[:47] + "..."
            print(f"{status} {row['question_number']}: {details_str}")
        
        return df
    
    def run_benchmark(self, max_questions: int = 3) -> pd.DataFrame:
        """Run the benchmark on the first few questions."""
        # Get all question directories
        question_dirs = [d for d in os.listdir(self.dataset_path) 
                        if os.path.isdir(os.path.join(self.dataset_path, d)) 
                        and d.startswith('q')]
        
        # Sort by question number
        question_dirs.sort(key=lambda x: int(x[1:]))
        
        # Limit to first max_questions
        question_dirs = question_dirs[:max_questions]
        print(f"Processing first {len(question_dirs)} questions: {question_dirs}")
        
        start_time = time.time()
        
        for i, q_dir in enumerate(question_dirs):
            print(f"\n{'='*70}")
            print(f"QUESTION {q_dir} ({i+1}/{len(question_dirs)})")
            print(f"Elapsed time: {time.time() - start_time:.1f} seconds")
            print(f"{'='*70}")
            
            try:
                # Load question data
                question_data = self.load_question_data(q_dir)
                print(f"Question: {question_data['question']}")
                
                # Evaluate question
                result = self.evaluate_question(question_data, q_dir)
                result['question_number'] = q_dir
                result['question_text'] = question_data['question']
                result['processing_time'] = time.time() - start_time
                
                self.results.append(result)
                
                ic = result['is_correct']
                status = "✓ CORRECT" if ic is True else ("– SKIPPED" if pd.isna(ic) else "✗ INCORRECT")
                print(f"\n{status} - {q_dir}")
                
            except Exception as e:
                print(f"✗ ERROR processing {q_dir}: {str(e)}")
                self.results.append({
                    'question_number': q_dir,
                    'error': str(e),
                    'is_correct': False,
                    'processing_time': time.time() - start_time
                })
            
            # Save intermediate results every 5 questions
            if (i + 1) % 5 == 0:
                temp_df = pd.DataFrame(self.results)
                temp_df.to_csv(f'results/temp_results_{i+1}.csv', index=False)
                print(f"Intermediate results saved to temp_results_{i+1}.csv")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate metrics (exclude skipped/None from accuracy)
        total_questions = len(df)
        evaluated = df['is_correct'].notna().sum() if 'is_correct' in df.columns else 0
        correct_answers = (df['is_correct'] == True).sum() if 'is_correct' in df.columns else 0
        accuracy = (correct_answers / evaluated) if evaluated > 0 else 0
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Total Questions: {total_questions}")
        print(f"Evaluated (excl. skipped): {evaluated}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Total Time: {total_time:.1f} seconds")
        print(f"Average Time per Question: {total_time/total_questions:.1f} seconds")
        
        # Grouped accuracy by question type (transfer vs other)
        if 'question_text' in df.columns:
            try:
                df['is_transfer'] = df['question_text'].str.contains(r'transfer function', case=False, na=False)
                for group_name, group_mask in [("TRANSFER GROUP", df['is_transfer'] == True), ("OTHER GROUP", df['is_transfer'] == False)]:
                    gdf = df[group_mask]
                    g_total = len(gdf)
                    g_evaluated = gdf['is_correct'].notna().sum()
                    g_correct = (gdf['is_correct'] == True).sum()
                    g_accuracy = (g_correct / g_evaluated) if g_evaluated > 0 else 0
                    print(f"\n[{group_name}]")
                    print(f"  Total: {g_total}")
                    print(f"  Evaluated: {g_evaluated}")
                    print(f"  Correct: {g_correct}")
                    print(f"  Accuracy: {g_accuracy:.2%}")
            except Exception as e:
                print(f"Warning: failed to compute grouped accuracy: {str(e)}")
        
        # Show detailed results
        print(f"\nDetailed Results:")
        print("-"*70)
        for _, row in df.iterrows():
            ic = row.get('is_correct', None)
            status = "✓" if ic is True else ("–" if pd.isna(ic) else "✗")
            details_val = row.get('comparison_details', None)
            if details_val is None or (isinstance(details_val, float) and pd.isna(details_val)):
                details_str = 'N/A'
            else:
                details_str = str(details_val)
            if len(details_str) > 50:
                details_str = details_str[:47] + "..."
            print(f"{status} {row['question_number']}: {details_str}")
        
        return df

if __name__ == "__main__":
    import sys
    dataset_path = "./symbolic_level0_dataset"

    # Check command line arguments for mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    max_questions = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 600
    if len(sys.argv) > 3:
        dataset_path = sys.argv[3]

    # Defer model initialization for evaluation-only mode to avoid requiring provider SDKs
    if mode == "evaluation":
        benchmark = SymbolicEquationBenchmark(dataset_path, init_model=False)
    else:
        benchmark = SymbolicEquationBenchmark(dataset_path)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if mode == "inference":
        results_df = benchmark.run_inference_only(max_questions=max_questions)
        output_file = f'results/inference_gemini25_symbolic_level15_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nInference results saved to {output_file}")
        
    elif mode == "evaluation":
        results_df = benchmark.run_evaluation_only(max_questions=max_questions)
        output_file = f'results/evaluation_gemini25_symbolic_level15_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)

    else:  
        results_df = benchmark.run_benchmark(max_questions=max_questions)
        output_file = f'results/results_gemini25_symbolic_level15_v2_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)

