#!/usr/bin/env python3
"""
Pipeline Output Authenticity Audit

Comprehensive audit to identify mock data, fake results, and non-functional outputs
in the autonomous persona agent pipeline. This audit ensures the system produces
genuine, functional results rather than masking failures with mock data.

Audit Categories:
1. LLM Response Authenticity
2. Template Generation Verification
3. POML Template Validity
4. Agent Decision-Making Authenticity
5. Simulation Result Verification
6. Performance Metrics Validation
7. Error Handling Authenticity
8. Mock Data Detection Patterns
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MockDataEvidence:
    """Evidence of potential mock data usage"""

    location: str
    evidence_type: str
    description: str
    severity: str
    sample_data: str
    confidence: float


class OutputAuthenticityAuditor:
    """
    Comprehensive auditor to detect mock data and verify authentic outputs
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.evidence = []
        self.audit_results = {
            "mock_data_detected": [],
            "suspicious_patterns": [],
            "hardcoded_responses": [],
            "non_functional_outputs": [],
            "validation_failures": [],
        }

        # Common mock data patterns
        self.mock_patterns = {
            "generic_responses": [
                r"mock\s+response",
                r"placeholder\s+text",
                r"lorem ipsum",
                r"test\s+output",
                r"sample\s+data",
                r"dummy\s+content",
                r"fake\s+result",
                r"generated\s+by\s+agent\s+\d+",
                r"mock\s+\w+\s+\d+",
            ],
            "hardcoded_values": [
                r'return\s+"[^"]*mock[^"]*"',
                r'content\s*=\s*"[^"]*test[^"]*"',
                r'response\s*=\s*"[^"]*placeholder[^"]*"',
                r"Mock\(\w*\)",
                r"AsyncMock\(\w*\)",
                r"\.return_value\s*=",
                r"should_fail\s*=\s*False",
            ],
            "suspicious_consistency": [
                r"response\s+\d+\s+for\s+persona",
                r"template_\d+\.poml",
                r"simulation_id_\d+",
                r"character_\d+",
                r"scene_\d+",
            ],
            "non_functional_markers": [
                r"# TODO:.*implement",
                r"raise NotImplementedError",
                r"pass\s*#.*placeholder",
                r"mock.*implementation",
                r"fallback.*mock",
            ],
        }

    async def audit_pipeline_outputs(self) -> Dict[str, Any]:
        """
        Run comprehensive audit of all pipeline outputs
        """
        logger.info("Starting pipeline output authenticity audit...")

        audit_methods = [
            self.audit_llm_response_authenticity,
            self.audit_template_generation_validity,
            self.audit_poml_template_functionality,
            self.audit_agent_decision_authenticity,
            self.audit_simulation_result_validity,
            self.audit_performance_metrics_authenticity,
            self.audit_error_handling_genuineness,
            self.audit_codebase_for_mock_patterns,
        ]

        for audit_method in audit_methods:
            try:
                logger.info(f"Running {audit_method.__name__}...")
                await audit_method()
            except Exception as e:
                logger.error(f"Audit method {audit_method.__name__} failed: {e}")
                self.add_evidence(
                    location=audit_method.__name__,
                    evidence_type="audit_failure",
                    description=f"Audit method crashed: {e}",
                    severity="high",
                    sample_data=str(e)[:200],
                    confidence=1.0,
                )

        return self.generate_authenticity_report()

    async def audit_llm_response_authenticity(self):
        """
        Audit LLM responses for authentic vs mock content
        """
        logger.info("Auditing LLM response authenticity...")

        # Check for mock LLM orchestrator usage
        orchestrator_files = list(self.project_root.glob("**/orchestrat*.py"))

        for file_path in orchestrator_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                # Look for mock implementations
                if "Mock" in content and "LLMOrchestrator" in content:
                    # Check if it's test code vs production code
                    if "test" not in str(file_path).lower():
                        self.add_evidence(
                            location=str(file_path),
                            evidence_type="mock_llm_in_production",
                            description="Mock LLM orchestrator found in production code",
                            severity="critical",
                            sample_data=self.extract_code_sample(content, "Mock"),
                            confidence=0.9,
                        )

                # Look for hardcoded responses
                for pattern in self.mock_patterns["hardcoded_values"]:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        self.add_evidence(
                            location=f"{file_path}:{self.get_line_number(content, match.start())}",
                            evidence_type="hardcoded_response",
                            description="Hardcoded response pattern detected",
                            severity="high",
                            sample_data=match.group(0),
                            confidence=0.8,
                        )

                # Check for suspicious response patterns
                if "generate_with_persona" in content:
                    # Look for implementations that don't actually call LLM
                    if "return Mock" in content or "AsyncMock" in content:
                        self.add_evidence(
                            location=str(file_path),
                            evidence_type="mock_llm_implementation",
                            description="LLM generation method returns mock data",
                            severity="critical",
                            sample_data=self.extract_code_sample(
                                content, "generate_with_persona"
                            ),
                            confidence=0.95,
                        )

            except Exception as e:
                logger.error(f"Error auditing {file_path}: {e}")

    async def audit_template_generation_validity(self):
        """
        Audit template generation for authentic vs pre-written templates
        """
        logger.info("Auditing template generation validity...")

        # Find relevant agent files
        agent_files = list(self.project_root.glob("**/autonomous_persona_agents.py"))

        for file_path in agent_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                # Look for template generation methods
                if "generate_base_template" in content:
                    # Check if template generation is actually dynamic
                    template_method_start = content.find("def generate_base_template")
                    if template_method_start != -1:
                        # Extract method content
                        method_content = self.extract_method_content(
                            content, template_method_start
                        )

                        # Check for static template returns
                        if (
                            'return """' in method_content
                            or "return '''" in method_content
                        ):
                            self.add_evidence(
                                location=f"{file_path}:generate_base_template",
                                evidence_type="static_template_generation",
                                description="Template generation returns static content",
                                severity="high",
                                sample_data=method_content[:200],
                                confidence=0.85,
                            )

                        # Check for template loading from files vs generation
                        if (
                            ".read_text()" in method_content
                            and "generate" not in method_content.lower()
                        ):
                            self.add_evidence(
                                location=f"{file_path}:generate_base_template",
                                evidence_type="template_file_loading",
                                description="Template 'generation' just loads pre-existing files",
                                severity="medium",
                                sample_data=method_content[:200],
                                confidence=0.7,
                            )

                # Look for modify_template methods
                if "modify_template" in content:
                    modify_method_start = content.find("def modify_template")
                    if modify_method_start != -1:
                        method_content = self.extract_method_content(
                            content, modify_method_start
                        )

                        # Check if modification is actually happening
                        if "return existing_template" in method_content:
                            self.add_evidence(
                                location=f"{file_path}:modify_template",
                                evidence_type="non_functional_modification",
                                description="Template modification returns unchanged template",
                                severity="high",
                                sample_data=method_content[:200],
                                confidence=0.9,
                            )

            except Exception as e:
                logger.error(f"Error auditing template generation in {file_path}: {e}")

        # Check existing template files for authenticity
        poml_files = list(self.project_root.glob("**/*.poml"))

        for template_file in poml_files:
            try:
                content = template_file.read_text(encoding="utf-8")

                # Check for placeholder content
                for pattern in self.mock_patterns["generic_responses"]:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.add_evidence(
                            location=str(template_file),
                            evidence_type="placeholder_template_content",
                            description="Template contains placeholder/mock content",
                            severity="medium",
                            sample_data=content[:200],
                            confidence=0.75,
                        )

            except Exception as e:
                logger.error(f"Error auditing template file {template_file}: {e}")

    async def audit_poml_template_functionality(self):
        """
        Audit POML templates for functional vs non-functional content
        """
        logger.info("Auditing POML template functionality...")

        poml_files = list(self.project_root.glob("**/*.poml"))

        for template_file in poml_files:
            try:
                content = template_file.read_text(encoding="utf-8")

                # Validate XML structure
                try:
                    if content.strip():
                        # Try to parse as XML
                        ET.fromstring(content)
                        template_is_valid_xml = True
                    else:
                        template_is_valid_xml = False

                except ET.ParseError:
                    self.add_evidence(
                        location=str(template_file),
                        evidence_type="invalid_poml_structure",
                        description="POML template has invalid XML structure",
                        severity="high",
                        sample_data=content[:200],
                        confidence=1.0,
                    )
                    template_is_valid_xml = False

                if template_is_valid_xml:
                    # Check for functional template elements
                    has_system_prompt = "<system>" in content
                    has_user_prompt = "<user>" in content

                    if not (has_system_prompt or has_user_prompt):
                        self.add_evidence(
                            location=str(template_file),
                            evidence_type="non_functional_template",
                            description="POML template lacks system/user prompt structure",
                            severity="medium",
                            sample_data=content[:200],
                            confidence=0.8,
                        )

                    # Check for mock content in prompts
                    if has_system_prompt or has_user_prompt:
                        for pattern in self.mock_patterns["generic_responses"]:
                            if re.search(pattern, content, re.IGNORECASE):
                                self.add_evidence(
                                    location=str(template_file),
                                    evidence_type="mock_prompt_content",
                                    description="POML template contains mock/placeholder prompts",
                                    severity="medium",
                                    sample_data=content[:200],
                                    confidence=0.75,
                                )
                                break

                # Check template size and complexity
                if len(content) < 50:
                    self.add_evidence(
                        location=str(template_file),
                        evidence_type="minimal_template_content",
                        description="POML template is suspiciously minimal",
                        severity="low",
                        sample_data=content,
                        confidence=0.6,
                    )

            except Exception as e:
                logger.error(f"Error auditing POML template {template_file}: {e}")

    async def audit_agent_decision_authenticity(self):
        """
        Audit agent decision-making for genuine vs predetermined responses
        """
        logger.info("Auditing agent decision authenticity...")

        agent_files = list(self.project_root.glob("**/autonomous_persona_agents.py"))

        for file_path in agent_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                # Look for run_simulation methods
                if "def run_simulation" in content:
                    simulation_methods = re.finditer(
                        r"def run_simulation.*?(?=def|\Z)", content, re.DOTALL
                    )

                    for match in simulation_methods:
                        method_content = match.group(0)

                        # Check for predetermined decision patterns
                        if "if context_data.get(" in method_content:
                            # Good - context-based decisions
                            pass
                        elif (
                            "return {" in method_content
                            and "await" not in method_content
                        ):
                            # Suspicious - returning static results without LLM interaction
                            self.add_evidence(
                                location=f"{file_path}:run_simulation",
                                evidence_type="static_simulation_results",
                                description="Simulation returns static results without LLM interaction",
                                severity="high",
                                sample_data=method_content[:300],
                                confidence=0.8,
                            )

                        # Check for mock decision patterns
                        for pattern in self.mock_patterns["suspicious_consistency"]:
                            if re.search(pattern, method_content):
                                self.add_evidence(
                                    location=f"{file_path}:run_simulation",
                                    evidence_type="suspicious_decision_pattern",
                                    description="Agent decisions follow suspicious mock patterns",
                                    severity="medium",
                                    sample_data=method_content[:200],
                                    confidence=0.7,
                                )

                # Look for template selection logic
                if "select_template" in content or "choose_template" in content:
                    # Check if template selection is deterministic vs intelligent
                    template_selection_pattern = (
                        r"(select_template|choose_template).*?(?=def|\Z)"
                    )
                    matches = re.finditer(
                        template_selection_pattern, content, re.DOTALL
                    )

                    for match in matches:
                        method_content = match.group(0)
                        if (
                            "random" in method_content
                            and "context" not in method_content
                        ):
                            self.add_evidence(
                                location=f"{file_path}:template_selection",
                                evidence_type="random_template_selection",
                                description="Template selection is random rather than context-aware",
                                severity="medium",
                                sample_data=method_content[:200],
                                confidence=0.75,
                            )

            except Exception as e:
                logger.error(f"Error auditing agent decisions in {file_path}: {e}")

    async def audit_simulation_result_validity(self):
        """
        Audit simulation results for genuine vs fake outputs
        """
        logger.info("Auditing simulation result validity...")

        # Look for example files and demonstrations
        example_files = list(self.project_root.glob("**/example*.py")) + list(
            self.project_root.glob("**/demo*.py")
        )

        for file_path in example_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                # Check for hardcoded simulation results
                if "workflow_results" in content:
                    # Look for the results definition
                    results_start = content.find("workflow_results")
                    if results_start != -1:
                        results_section = content[results_start : results_start + 2000]

                        # Check if results are hardcoded vs computed
                        if (
                            "'stages_completed': [" in results_section
                            and "await" not in results_section
                        ):
                            self.add_evidence(
                                location=f"{file_path}:workflow_results",
                                evidence_type="hardcoded_simulation_results",
                                description="Simulation results are hardcoded rather than computed",
                                severity="critical",
                                sample_data=results_section[:300],
                                confidence=0.95,
                            )

                # Look for mocked performance metrics
                if "metrics" in content and "total_simulations" in content:
                    metrics_pattern = r"metrics.*?=.*?\{.*?\}"
                    matches = re.finditer(metrics_pattern, content, re.DOTALL)

                    for match in matches:
                        metrics_content = match.group(0)
                        if (
                            re.search(r"success_rate.*0\.", metrics_content)
                            or re.search(r"total_simulations.*\d+", metrics_content)
                        ):
                            # Check if metrics are static
                            if (
                                "calculate" not in metrics_content
                                and "compute" not in metrics_content
                            ):
                                self.add_evidence(
                                    location=f"{file_path}:metrics",
                                    evidence_type="static_performance_metrics",
                                    description="Performance metrics appear to be static/hardcoded",
                                    severity="high",
                                    sample_data=metrics_content[:200],
                                    confidence=0.8,
                                )

                # Check for simulation execution patterns
                if "await.*simulation" in content:
                    # Good - async simulation execution
                    pass
                elif "simulation.*=" in content and "mock" not in content.lower():
                    # Check if simulations are actually executed or just assigned
                    simulation_pattern = r"simulation.*?=.*?(?=\n|\r)"
                    matches = re.finditer(simulation_pattern, content)

                    for match in matches:
                        sim_line = match.group(0)
                        if (
                            "{" in sim_line
                            and "}" in sim_line
                            and "await" not in sim_line
                        ):
                            self.add_evidence(
                                location=f"{file_path}:{self.get_line_number(content, match.start())}",
                                evidence_type="non_executed_simulation",
                                description="Simulation results assigned without execution",
                                severity="medium",
                                sample_data=sim_line,
                                confidence=0.7,
                            )

            except Exception as e:
                logger.error(f"Error auditing simulation results in {file_path}: {e}")

    async def audit_performance_metrics_authenticity(self):
        """
        Audit performance metrics for authentic vs fake measurements
        """
        logger.info("Auditing performance metrics authenticity...")

        # Look for orchestrator and engine files
        metric_files = (
            list(self.project_root.glob("**/*orchestrat*.py"))
            + list(self.project_root.glob("**/*engine*.py"))
            + list(self.project_root.glob("**/*metrics*.py"))
        )

        for file_path in metric_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                # Look for performance tracking methods
                if "get_performance" in content or "performance_summary" in content:
                    perf_methods = re.finditer(
                        r"def.*performance.*?(?=def|\Z)", content, re.DOTALL
                    )

                    for match in perf_methods:
                        method_content = match.group(0)

                        # Check if metrics are computed vs hardcoded
                        if "return {" in method_content:
                            # Look for actual computation
                            has_computation = any(
                                op in method_content
                                for op in [
                                    "len(",
                                    "sum(",
                                    "count",
                                    "time()",
                                    "statistics",
                                ]
                            )
                            has_tracking = any(
                                track in method_content
                                for track in [
                                    "self.metrics",
                                    "self.stats",
                                    "self.counters",
                                ]
                            )

                            if not (has_computation or has_tracking):
                                self.add_evidence(
                                    location=f"{file_path}:performance_method",
                                    evidence_type="non_computed_metrics",
                                    description="Performance metrics returned without computation or tracking",
                                    severity="high",
                                    sample_data=method_content[:300],
                                    confidence=0.85,
                                )

                # Look for timing measurements
                if "response_time" in content or "execution_time" in content:
                    # Check for actual timing vs mock timing
                    if "time.time()" not in content and "perf_counter()" not in content:
                        if "response_time.*=" in content:
                            self.add_evidence(
                                location=str(file_path),
                                evidence_type="mock_timing_metrics",
                                description="Response time metrics without actual timing measurement",
                                severity="medium",
                                sample_data=self.extract_code_sample(
                                    content, "response_time"
                                ),
                                confidence=0.75,
                            )

                # Look for success rate calculations
                if "success_rate" in content:
                    success_pattern = r"success_rate.*?=.*?(?=\n|\r)"
                    matches = re.finditer(success_pattern, content)

                    for match in matches:
                        success_line = match.group(0)
                        if (
                            "0." in success_line or "1.0" in success_line
                        ) and "/" not in success_line:
                            self.add_evidence(
                                location=f"{file_path}:{self.get_line_number(content, match.start())}",
                                evidence_type="hardcoded_success_rate",
                                description="Success rate appears hardcoded rather than calculated",
                                severity="medium",
                                sample_data=success_line,
                                confidence=0.7,
                            )

            except Exception as e:
                logger.error(f"Error auditing performance metrics in {file_path}: {e}")

    async def audit_error_handling_genuineness(self):
        """
        Audit error handling for genuine vs suppressed errors
        """
        logger.info("Auditing error handling genuineness...")

        all_python_files = list(self.project_root.glob("**/*.py"))

        for file_path in all_python_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                # Look for suspicious error handling patterns
                except_blocks = re.finditer(
                    r"except.*?:.*?(?=except|finally|def|class|\Z)", content, re.DOTALL
                )

                for match in except_blocks:
                    except_content = match.group(0)

                    # Check for error suppression
                    if "pass" in except_content and "log" not in except_content:
                        self.add_evidence(
                            location=f"{file_path}:{self.get_line_number(content, match.start())}",
                            evidence_type="error_suppression",
                            description="Error caught but suppressed without logging",
                            severity="medium",
                            sample_data=except_content.strip()[:200],
                            confidence=0.8,
                        )

                    # Check for mock error responses
                    if "return" in except_content and any(
                        mock in except_content
                        for mock in ["mock", "fake", "placeholder"]
                    ):
                        self.add_evidence(
                            location=f"{file_path}:{self.get_line_number(content, match.start())}",
                            evidence_type="mock_error_response",
                            description="Error handler returns mock/fake response",
                            severity="high",
                            sample_data=except_content.strip()[:200],
                            confidence=0.85,
                        )

                # Look for fallback mechanisms that might hide failures
                if "fallback" in content:
                    fallback_pattern = r"fallback.*?(?=\n|\r)"
                    matches = re.finditer(fallback_pattern, content, re.IGNORECASE)

                    for match in matches:
                        fallback_line = match.group(0)
                        if any(
                            word in fallback_line.lower()
                            for word in ["mock", "default", "placeholder"]
                        ):
                            self.add_evidence(
                                location=f"{file_path}:{self.get_line_number(content, match.start())}",
                                evidence_type="mock_fallback_mechanism",
                                description="Fallback mechanism uses mock/default responses",
                                severity="medium",
                                sample_data=fallback_line,
                                confidence=0.75,
                            )

            except Exception as e:
                logger.error(f"Error auditing error handling in {file_path}: {e}")

    async def audit_codebase_for_mock_patterns(self):
        """
        Comprehensive scan for mock data patterns across codebase
        """
        logger.info("Auditing codebase for mock patterns...")

        all_files = (
            list(self.project_root.glob("**/*.py"))
            + list(self.project_root.glob("**/*.poml"))
            + list(self.project_root.glob("**/*.md"))
        )

        for file_path in all_files:
            try:
                # Skip test files for mock pattern detection
                if "test" in str(file_path).lower():
                    continue

                content = file_path.read_text(encoding="utf-8")

                # Check all mock pattern categories
                for category, patterns in self.mock_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(
                            pattern, content, re.IGNORECASE | re.MULTILINE
                        )

                        for match in matches:
                            self.add_evidence(
                                location=f"{file_path}:{self.get_line_number(content, match.start())}",
                                evidence_type=f"mock_pattern_{category}",
                                description=f"Mock pattern detected: {pattern}",
                                severity=(
                                    "medium"
                                    if category
                                    in ["suspicious_consistency", "generic_responses"]
                                    else "high"
                                ),
                                sample_data=match.group(0),
                                confidence=0.7,
                            )

                # Look for specific anti-patterns
                anti_patterns = [
                    r"# Note:.*mock.*implementation",
                    r"# TODO:.*replace.*mock",
                    r"# Uncomment.*run.*with.*real",
                    r"For now.*simulated.*version",
                ]

                for pattern in anti_patterns:
                    matches = re.finditer(
                        pattern, content, re.IGNORECASE | re.MULTILINE
                    )

                    for match in matches:
                        self.add_evidence(
                            location=f"{file_path}:{self.get_line_number(content, match.start())}",
                            evidence_type="mock_implementation_indicator",
                            description="Comment indicates mock/placeholder implementation",
                            severity="high",
                            sample_data=match.group(0),
                            confidence=0.9,
                        )

            except Exception as e:
                logger.error(f"Error scanning {file_path} for mock patterns: {e}")

    # Utility methods

    def add_evidence(
        self,
        location: str,
        evidence_type: str,
        description: str,
        severity: str,
        sample_data: str,
        confidence: float,
    ):
        """Add evidence to the audit results"""

        evidence = MockDataEvidence(
            location=location,
            evidence_type=evidence_type,
            description=description,
            severity=severity,
            sample_data=sample_data,
            confidence=confidence,
        )

        self.evidence.append(evidence)

        # Categorize by type
        if evidence_type.startswith("mock_"):
            self.audit_results["mock_data_detected"].append(evidence)
        elif evidence_type.startswith("hardcoded_"):
            self.audit_results["hardcoded_responses"].append(evidence)
        elif evidence_type.startswith("non_functional_"):
            self.audit_results["non_functional_outputs"].append(evidence)
        elif evidence_type.startswith("invalid_"):
            self.audit_results["validation_failures"].append(evidence)
        else:
            self.audit_results["suspicious_patterns"].append(evidence)

    def get_line_number(self, content: str, position: int) -> int:
        """Get line number for a position in content"""
        return content[:position].count("\n") + 1

    def extract_code_sample(
        self, content: str, keyword: str, context_lines: int = 2
    ) -> str:
        """Extract code sample around keyword"""
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if keyword in line:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                return "\n".join(lines[start:end])

        return keyword  # Fallback

    def extract_method_content(self, content: str, method_start: int) -> str:
        """Extract method content from start position"""
        lines = content[method_start:].split("\n")
        method_lines = []
        indent_level = None

        for line in lines:
            if line.strip().startswith("def "):
                method_lines.append(line)
                # Determine base indent
                indent_level = len(line) - len(line.lstrip())
            elif line.strip() == "":
                method_lines.append(line)
            elif indent_level is not None:
                current_indent = len(line) - len(line.lstrip())
                if current_indent > indent_level or line.strip() == "":
                    method_lines.append(line)
                else:
                    break
            else:
                method_lines.append(line)

            # Stop at reasonable length
            if len(method_lines) > 50:
                break

        return "\n".join(method_lines)

    def generate_authenticity_report(self) -> Dict[str, Any]:
        """Generate comprehensive authenticity audit report"""

        # Categorize evidence by severity
        critical_evidence = [e for e in self.evidence if e.severity == "critical"]
        high_evidence = [e for e in self.evidence if e.severity == "high"]
        medium_evidence = [e for e in self.evidence if e.severity == "medium"]
        low_evidence = [e for e in self.evidence if e.severity == "low"]

        # Calculate authenticity score
        total_evidence = len(self.evidence)
        severity_weights = {"critical": 10, "high": 5, "medium": 2, "low": 1}

        weighted_score = sum(severity_weights[e.severity] for e in self.evidence)
        max_possible_score = total_evidence * severity_weights["critical"]

        authenticity_score = max(
            0, 100 - (weighted_score / max(1, max_possible_score) * 100)
        )

        # Identify top concerns
        top_concerns = sorted(
            self.evidence,
            key=lambda e: (severity_weights[e.severity], e.confidence),
            reverse=True,
        )[:10]

        # Generate recommendations
        recommendations = []

        if critical_evidence:
            recommendations.append(
                {
                    "priority": "CRITICAL",
                    "issue": "Mock data in production pipeline",
                    "recommendation": "Remove all mock implementations from production code",
                    "count": len(critical_evidence),
                }
            )

        if high_evidence:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "issue": "Hardcoded responses and non-functional outputs",
                    "recommendation": "Implement genuine LLM integration and dynamic responses",
                    "count": len(high_evidence),
                }
            )

        if medium_evidence or low_evidence:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "issue": "Suspicious patterns and placeholder content",
                    "recommendation": "Review and enhance output authenticity",
                    "count": len(medium_evidence) + len(low_evidence),
                }
            )

        report = {
            "summary": {
                "total_evidence_found": total_evidence,
                "authenticity_score": authenticity_score,
                "critical_issues": len(critical_evidence),
                "high_issues": len(high_evidence),
                "medium_issues": len(medium_evidence),
                "low_issues": len(low_evidence),
                "pipeline_status": self.determine_pipeline_status(
                    authenticity_score, critical_evidence
                ),
            },
            "evidence_breakdown": {
                "mock_data_detected": len(self.audit_results["mock_data_detected"]),
                "hardcoded_responses": len(self.audit_results["hardcoded_responses"]),
                "non_functional_outputs": len(
                    self.audit_results["non_functional_outputs"]
                ),
                "validation_failures": len(self.audit_results["validation_failures"]),
                "suspicious_patterns": len(self.audit_results["suspicious_patterns"]),
            },
            "top_concerns": [
                {
                    "location": e.location,
                    "type": e.evidence_type,
                    "description": e.description,
                    "severity": e.severity,
                    "confidence": e.confidence,
                    "sample": e.sample_data[:100],
                }
                for e in top_concerns
            ],
            "recommendations": recommendations,
            "detailed_evidence": [
                {
                    "location": e.location,
                    "evidence_type": e.evidence_type,
                    "description": e.description,
                    "severity": e.severity,
                    "sample_data": e.sample_data,
                    "confidence": e.confidence,
                }
                for e in self.evidence
            ],
        }

        return report

    def determine_pipeline_status(
        self, authenticity_score: float, critical_evidence: List
    ) -> str:
        """Determine overall pipeline authenticity status"""

        if critical_evidence:
            return "COMPROMISED"
        elif authenticity_score < 50:
            return "HIGH_MOCK_DATA"
        elif authenticity_score < 75:
            return "MODERATE_CONCERNS"
        elif authenticity_score < 90:
            return "MINOR_ISSUES"
        else:
            return "AUTHENTIC"


async def run_output_authenticity_audit(project_root: str = ".") -> Dict[str, Any]:
    """
    Run comprehensive output authenticity audit
    """
    print("\n" + "=" * 80)
    print("🕵️ AUTONOMOUS AGENT PIPELINE OUTPUT AUTHENTICITY AUDIT")
    print("=" * 80)
    print("Scanning for mock data, fake results, and non-functional outputs...")

    auditor = OutputAuthenticityAuditor(project_root)
    report = await auditor.audit_pipeline_outputs()

    summary = report["summary"]

    print("\n📊 AUDIT SUMMARY:")
    print(f"   Pipeline Status: {summary['pipeline_status']}")
    print(f"   Authenticity Score: {summary['authenticity_score']:.1f}/100")
    print(f"   Total Evidence Found: {summary['total_evidence_found']}")

    print("\n🚨 ISSUE BREAKDOWN:")
    print(f"   🔴 Critical: {summary['critical_issues']} (Mock data in production)")
    print(f"   🟡 High: {summary['high_issues']} (Hardcoded/non-functional outputs)")
    print(f"   🟠 Medium: {summary['medium_issues']} (Suspicious patterns)")
    print(f"   🟢 Low: {summary['low_issues']} (Minor concerns)")

    print("\n🔍 EVIDENCE BREAKDOWN:")
    breakdown = report["evidence_breakdown"]
    print(f"   Mock Data Detected: {breakdown['mock_data_detected']}")
    print(f"   Hardcoded Responses: {breakdown['hardcoded_responses']}")
    print(f"   Non-functional Outputs: {breakdown['non_functional_outputs']}")
    print(f"   Validation Failures: {breakdown['validation_failures']}")
    print(f"   Suspicious Patterns: {breakdown['suspicious_patterns']}")

    print("\n🎯 TOP CONCERNS:")
    for i, concern in enumerate(report["top_concerns"][:5], 1):
        severity_icon = {
            "critical": "🔴",
            "high": "🟡",
            "medium": "🟠",
            "low": "🟢",
        }.get(concern["severity"], "⚪")
        print(
            f"   {i}. {severity_icon} [{concern['severity'].upper()}] {concern['description']}"
        )
        print(f"      Location: {concern['location']}")
        print(f"      Sample: {concern['sample'][:50]}...")

    print("\n🔧 RECOMMENDATIONS:")
    for rec in report["recommendations"]:
        priority_icon = {"CRITICAL": "🔴", "HIGH": "🟡", "MEDIUM": "🟠"}.get(
            rec["priority"], "🟢"
        )
        print(
            f"   {priority_icon} [{rec['priority']}] {rec['issue']} ({rec['count']} issues)"
        )
        print(f"      → {rec['recommendation']}")

    status_messages = {
        "COMPROMISED": "🔴 CRITICAL: Pipeline heavily relies on mock data - not production ready",
        "HIGH_MOCK_DATA": "🟡 WARNING: Significant mock data usage detected",
        "MODERATE_CONCERNS": "🟠 CAUTION: Some authenticity concerns identified",
        "MINOR_ISSUES": "🟢 GOOD: Minor issues, mostly production ready",
        "AUTHENTIC": "✅ EXCELLENT: Pipeline outputs appear authentic",
    }

    print("\n📈 OVERALL ASSESSMENT:")
    print(f"   {status_messages.get(summary['pipeline_status'], 'Unknown status')}")

    if summary["authenticity_score"] >= 90:
        print("   ✅ Pipeline outputs are highly authentic")
    elif summary["authenticity_score"] >= 75:
        print("   ⚠️ Pipeline has moderate authenticity issues")
    else:
        print("   🔴 Pipeline authenticity is compromised")

    return report


if __name__ == "__main__":
    import sys

    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    asyncio.run(run_output_authenticity_audit(project_root))
