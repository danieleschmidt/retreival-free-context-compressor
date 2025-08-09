"""Quality assurance framework with schema validation, testing, and performance profiling."""

import concurrent.futures
import cProfile
import io
import json
import logging
import pstats
import random
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import psutil
import torch

from .exceptions import ValidationError
from .monitoring_enhanced import get_distributed_tracer, get_enhanced_metrics_collector


logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Schema validation rule."""
    field_path: str
    rule_type: str  # required, type, range, pattern, custom
    constraint: Any
    message: str
    severity: str = "error"  # error, warning


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    field_results: dict[str, bool] = field(default_factory=dict)


@dataclass
class TestCase:
    """Test case definition."""
    test_id: str
    name: str
    description: str
    test_type: str  # unit, integration, stress, chaos
    input_data: dict[str, Any]
    expected_output: dict[str, Any] | None = None
    max_duration_ms: float | None = None
    success_criteria: list[Callable[[Any], bool]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of test execution."""
    test_id: str
    success: bool
    duration_ms: float
    output: Any | None = None
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    trace_id: str | None = None
    executed_at: datetime = field(default_factory=datetime.now)


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 30
    requests_per_second: float | None = None
    test_data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChaosTestConfig:
    """Configuration for chaos testing."""
    failure_types: list[str] = field(default_factory=lambda: [
        "memory_pressure", "cpu_spike", "network_delay", "disk_io"
    ])
    failure_probability: float = 0.1  # 10% chance of failure
    failure_duration_seconds: int = 30


class SchemaValidator:
    """Advanced schema validation for inputs and outputs."""

    def __init__(self):
        """Initialize schema validator."""
        self.schemas: dict[str, list[ValidationRule]] = {}
        self.custom_validators: dict[str, Callable] = {}
        self._setup_default_schemas()

        logger.info("Schema validator initialized")

    def _setup_default_schemas(self) -> None:
        """Set up default schemas for common data structures."""
        # Compression input schema
        compression_input_rules = [
            ValidationRule("text", "required", None, "Text input is required"),
            ValidationRule("text", "type", str, "Text must be a string"),
            ValidationRule("text", "custom", "non_empty_text", "Text cannot be empty"),
            ValidationRule("model_name", "type", str, "Model name must be a string", "warning"),
            ValidationRule("compression_ratio", "range", (1.0, 100.0), "Compression ratio must be between 1 and 100"),
            ValidationRule("chunk_size", "range", (32, 8192), "Chunk size must be between 32 and 8192"),
        ]
        self.schemas["compression_input"] = compression_input_rules

        # Compression output schema
        compression_output_rules = [
            ValidationRule("mega_tokens", "required", None, "Mega tokens are required"),
            ValidationRule("mega_tokens", "type", list, "Mega tokens must be a list"),
            ValidationRule("compression_ratio", "type", (int, float), "Compression ratio must be numeric"),
            ValidationRule("compression_ratio", "range", (1.0, None), "Compression ratio must be positive"),
            ValidationRule("processing_time", "type", (int, float), "Processing time must be numeric"),
            ValidationRule("processing_time", "range", (0.0, None), "Processing time must be non-negative"),
        ]
        self.schemas["compression_output"] = compression_output_rules

        # API response schema
        api_response_rules = [
            ValidationRule("status", "required", None, "Status is required"),
            ValidationRule("status", "pattern", r"^(success|error)$", "Status must be 'success' or 'error'"),
            ValidationRule("data", "required", None, "Data field is required for success responses"),
            ValidationRule("error", "type", dict, "Error must be a dictionary", "warning"),
        ]
        self.schemas["api_response"] = api_response_rules

        # Custom validators
        self.custom_validators["non_empty_text"] = lambda x: isinstance(x, str) and x.strip() != ""
        self.custom_validators["valid_model_name"] = lambda x: isinstance(x, str) and len(x) > 0 and "/" in x

    def register_schema(self, schema_name: str, rules: list[ValidationRule]) -> None:
        """Register a new schema.
        
        Args:
            schema_name: Name of the schema
            rules: List of validation rules
        """
        self.schemas[schema_name] = rules
        logger.info(f"Registered schema: {schema_name}")

    def register_custom_validator(self, name: str, validator: Callable[[Any], bool]) -> None:
        """Register a custom validation function.
        
        Args:
            name: Name of the validator
            validator: Function that returns True if valid
        """
        self.custom_validators[name] = validator

    def validate(self, data: dict[str, Any], schema_name: str) -> ValidationResult:
        """Validate data against a schema.
        
        Args:
            data: Data to validate
            schema_name: Name of the schema to use
            
        Returns:
            ValidationResult
        """
        if schema_name not in self.schemas:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown schema: {schema_name}"]
            )

        result = ValidationResult(valid=True)
        rules = self.schemas[schema_name]

        for rule in rules:
            field_valid = self._validate_rule(data, rule, result)
            result.field_results[rule.field_path] = field_valid

            if not field_valid and rule.severity == "error":
                result.valid = False

        return result

    def _validate_rule(self, data: dict[str, Any], rule: ValidationRule, result: ValidationResult) -> bool:
        """Validate a single rule.
        
        Args:
            data: Data to validate
            rule: Validation rule
            result: Result object to update
            
        Returns:
            True if rule passes
        """
        try:
            field_value = self._get_nested_value(data, rule.field_path)

            if rule.rule_type == "required":
                if field_value is None:
                    self._add_error(result, rule, "Field is required")
                    return False

            elif rule.rule_type == "type":
                if field_value is not None and not isinstance(field_value, rule.constraint):
                    self._add_error(result, rule, f"Field must be of type {rule.constraint}")
                    return False

            elif rule.rule_type == "range":
                if field_value is not None:
                    min_val, max_val = rule.constraint
                    if min_val is not None and field_value < min_val:
                        self._add_error(result, rule, f"Value {field_value} is below minimum {min_val}")
                        return False
                    if max_val is not None and field_value > max_val:
                        self._add_error(result, rule, f"Value {field_value} is above maximum {max_val}")
                        return False

            elif rule.rule_type == "pattern":
                if field_value is not None:
                    import re
                    if not re.match(rule.constraint, str(field_value)):
                        self._add_error(result, rule, f"Value does not match pattern {rule.constraint}")
                        return False

            elif rule.rule_type == "custom":
                if field_value is not None:
                    validator_name = rule.constraint
                    if validator_name in self.custom_validators:
                        validator = self.custom_validators[validator_name]
                        if not validator(field_value):
                            self._add_error(result, rule, f"Custom validation failed: {validator_name}")
                            return False
                    else:
                        self._add_error(result, rule, f"Unknown custom validator: {validator_name}")
                        return False

            return True

        except Exception as e:
            self._add_error(result, rule, f"Validation error: {e}")
            return False

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _add_error(self, result: ValidationResult, rule: ValidationRule, message: str) -> None:
        """Add error or warning to result."""
        full_message = f"{rule.field_path}: {message}"

        if rule.severity == "error":
            result.errors.append(full_message)
        else:
            result.warnings.append(full_message)


class PerformanceProfiler:
    """Advanced performance profiler for memory and CPU optimization."""

    def __init__(self):
        """Initialize performance profiler."""
        self.profiles: dict[str, dict[str, Any]] = {}
        self.memory_snapshots: list[dict[str, Any]] = []
        self.active_profiles: dict[str, cProfile.Profile] = {}
        self._lock = threading.RLock()

        logger.info("Performance profiler initialized")

    @contextmanager
    def profile(self, operation_name: str, include_memory: bool = True):
        """Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation to profile
            include_memory: Whether to include memory profiling
        """
        profile_id = f"{operation_name}_{int(time.time() * 1000)}"

        # Start CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Get initial memory snapshot
        initial_memory = None
        if include_memory:
            initial_memory = self._get_memory_snapshot()

        start_time = time.time()

        with self._lock:
            self.active_profiles[profile_id] = profiler

        try:
            yield profile_id
        finally:
            # Stop profiling
            profiler.disable()
            end_time = time.time()

            # Get final memory snapshot
            final_memory = None
            if include_memory:
                final_memory = self._get_memory_snapshot()

            # Process results
            self._process_profile_results(
                profile_id, operation_name, profiler,
                start_time, end_time, initial_memory, final_memory
            )

            with self._lock:
                if profile_id in self.active_profiles:
                    del self.active_profiles[profile_id]

    def _get_memory_snapshot(self) -> dict[str, Any]:
        """Get current memory snapshot."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            snapshot = {
                "timestamp": time.time(),
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024
            }

            # Add GPU memory if available
            if torch.cuda.is_available():
                snapshot["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                snapshot["gpu_cached_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

            return snapshot

        except Exception as e:
            logger.error(f"Error getting memory snapshot: {e}")
            return {"error": str(e)}

    def _process_profile_results(
        self,
        profile_id: str,
        operation_name: str,
        profiler: cProfile.Profile,
        start_time: float,
        end_time: float,
        initial_memory: dict[str, Any] | None,
        final_memory: dict[str, Any] | None
    ) -> None:
        """Process profiling results."""
        duration = end_time - start_time

        # Get CPU profiling stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions

        # Calculate memory changes
        memory_delta = {}
        if initial_memory and final_memory:
            for key in initial_memory:
                if key in final_memory and isinstance(initial_memory[key], (int, float)):
                    memory_delta[f"{key}_delta"] = final_memory[key] - initial_memory[key]

        # Store results
        profile_result = {
            "profile_id": profile_id,
            "operation_name": operation_name,
            "duration_seconds": duration,
            "cpu_stats": stats_stream.getvalue(),
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "memory_delta": memory_delta,
            "timestamp": datetime.now()
        }

        with self._lock:
            self.profiles[profile_id] = profile_result

            # Keep only recent profiles
            if len(self.profiles) > 100:
                oldest_id = min(self.profiles.keys(),
                              key=lambda x: self.profiles[x]["timestamp"])
                del self.profiles[oldest_id]

        logger.info(f"Profile completed: {operation_name} in {duration:.3f}s")

    def get_profile_summary(self, operation_name: str | None = None) -> dict[str, Any]:
        """Get summary of profiling results.
        
        Args:
            operation_name: Filter by operation name
            
        Returns:
            Profile summary
        """
        with self._lock:
            profiles = list(self.profiles.values())

            if operation_name:
                profiles = [p for p in profiles if p["operation_name"] == operation_name]

            if not profiles:
                return {"message": "No profiles available"}

            durations = [p["duration_seconds"] for p in profiles]

            summary = {
                "total_profiles": len(profiles),
                "operation_filter": operation_name,
                "duration_stats": {
                    "mean": np.mean(durations),
                    "median": np.median(durations),
                    "min": np.min(durations),
                    "max": np.max(durations),
                    "std": np.std(durations)
                }
            }

            # Add memory stats if available
            memory_deltas = []
            for profile in profiles:
                if profile.get("memory_delta", {}).get("rss_mb_delta"):
                    memory_deltas.append(profile["memory_delta"]["rss_mb_delta"])

            if memory_deltas:
                summary["memory_stats"] = {
                    "mean_delta_mb": np.mean(memory_deltas),
                    "median_delta_mb": np.median(memory_deltas),
                    "max_delta_mb": np.max(memory_deltas)
                }

            return summary

    def export_profile(self, profile_id: str, format: str = "text") -> str:
        """Export profile results.
        
        Args:
            profile_id: Profile ID to export
            format: Export format ('text', 'json')
            
        Returns:
            Formatted profile data
        """
        with self._lock:
            if profile_id not in self.profiles:
                return f"Profile {profile_id} not found"

            profile = self.profiles[profile_id]

            if format == "json":
                return json.dumps(profile, indent=2, default=str)
            else:
                return f"""
Profile: {profile['operation_name']} ({profile['profile_id']})
Duration: {profile['duration_seconds']:.3f} seconds
Timestamp: {profile['timestamp']}

CPU Stats:
{profile['cpu_stats']}

Memory Delta:
{json.dumps(profile['memory_delta'], indent=2)}
"""


class TestFramework:
    """Comprehensive testing framework for integration, stress, and chaos testing."""

    def __init__(self):
        """Initialize test framework."""
        self.test_suites: dict[str, list[TestCase]] = {}
        self.test_results: list[TestResult] = []
        self.schema_validator = SchemaValidator()
        self.profiler = PerformanceProfiler()
        self._lock = threading.RLock()

        logger.info("Test framework initialized")

    def register_test_suite(self, suite_name: str, test_cases: list[TestCase]) -> None:
        """Register a test suite.
        
        Args:
            suite_name: Name of the test suite
            test_cases: List of test cases
        """
        with self._lock:
            self.test_suites[suite_name] = test_cases
            logger.info(f"Registered test suite '{suite_name}' with {len(test_cases)} test cases")

    def run_test_suite(
        self,
        suite_name: str,
        target_function: Callable,
        parallel: bool = False,
        max_workers: int | None = None
    ) -> list[TestResult]:
        """Run a test suite.
        
        Args:
            suite_name: Name of the test suite to run
            target_function: Function to test
            parallel: Whether to run tests in parallel
            max_workers: Maximum number of worker threads
            
        Returns:
            List of test results
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")

        test_cases = self.test_suites[suite_name]
        results = []

        if parallel and len(test_cases) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_test = {
                    executor.submit(self._run_single_test, test_case, target_function): test_case
                    for test_case in test_cases
                }

                for future in concurrent.futures.as_completed(future_to_test):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        test_case = future_to_test[future]
                        result = TestResult(
                            test_id=test_case.test_id,
                            success=False,
                            duration_ms=0,
                            error=str(e)
                        )
                        results.append(result)
        else:
            for test_case in test_cases:
                result = self._run_single_test(test_case, target_function)
                results.append(result)

        with self._lock:
            self.test_results.extend(results)

        # Log summary
        successful_tests = sum(1 for r in results if r.success)
        logger.info(f"Test suite '{suite_name}' completed: {successful_tests}/{len(results)} passed")

        return results

    def _run_single_test(self, test_case: TestCase, target_function: Callable) -> TestResult:
        """Run a single test case.
        
        Args:
            test_case: Test case to run
            target_function: Function to test
            
        Returns:
            Test result
        """
        tracer = get_distributed_tracer()
        metrics_collector = get_enhanced_metrics_collector()

        with tracer.span(f"test_{test_case.test_id}") as span:
            span.set_tag("test.id", test_case.test_id)
            span.set_tag("test.name", test_case.name)
            span.set_tag("test.type", test_case.test_type)

            start_time = time.time()

            try:
                # Run the test with profiling if it's a performance test
                with_profiling = test_case.test_type in ["stress", "performance"]

                if with_profiling:
                    with self.profiler.profile(f"test_{test_case.test_id}"):
                        output = target_function(**test_case.input_data)
                else:
                    output = target_function(**test_case.input_data)

                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                # Check timeout
                if test_case.max_duration_ms and duration_ms > test_case.max_duration_ms:
                    return TestResult(
                        test_id=test_case.test_id,
                        success=False,
                        duration_ms=duration_ms,
                        error=f"Test exceeded max duration: {duration_ms}ms > {test_case.max_duration_ms}ms",
                        trace_id=span.trace_id
                    )

                # Validate output schema if expected output is provided
                success = True
                error_msg = None

                if test_case.expected_output:
                    # Basic equality check (can be extended)
                    if output != test_case.expected_output:
                        success = False
                        error_msg = f"Output mismatch. Expected: {test_case.expected_output}, Got: {output}"

                # Run success criteria
                for criterion in test_case.success_criteria:
                    try:
                        if not criterion(output):
                            success = False
                            error_msg = f"Success criterion failed: {criterion.__name__}"
                            break
                    except Exception as e:
                        success = False
                        error_msg = f"Success criterion error: {e}"
                        break

                # Record metrics
                metrics = {
                    "duration_ms": duration_ms,
                    "success": 1 if success else 0
                }

                span.set_tag("test.success", success)
                span.set_tag("test.duration_ms", duration_ms)

                return TestResult(
                    test_id=test_case.test_id,
                    success=success,
                    duration_ms=duration_ms,
                    output=output,
                    error=error_msg,
                    metrics=metrics,
                    trace_id=span.trace_id
                )

            except Exception as e:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                span.set_error(e)
                span.set_tag("test.success", False)

                return TestResult(
                    test_id=test_case.test_id,
                    success=False,
                    duration_ms=duration_ms,
                    error=str(e),
                    trace_id=span.trace_id
                )

    def run_load_test(
        self,
        target_function: Callable,
        config: LoadTestConfig
    ) -> dict[str, Any]:
        """Run load/stress test.
        
        Args:
            target_function: Function to test
            config: Load test configuration
            
        Returns:
            Load test results
        """
        logger.info(f"Starting load test: {config.concurrent_users} users, {config.duration_seconds}s")

        results = []
        start_time = time.time()
        end_time = start_time + config.duration_seconds

        def worker():
            """Worker function for load testing."""
            while time.time() < end_time:
                try:
                    test_data = random.choice(config.test_data) if config.test_data else {}

                    request_start = time.time()
                    result = target_function(**test_data)
                    request_end = time.time()

                    results.append({
                        "success": True,
                        "duration_ms": (request_end - request_start) * 1000,
                        "timestamp": request_start
                    })

                    # Rate limiting
                    if config.requests_per_second:
                        time.sleep(1.0 / config.requests_per_second)

                except Exception as e:
                    request_end = time.time()
                    results.append({
                        "success": False,
                        "duration_ms": (request_end - request_start) * 1000 if 'request_start' in locals() else 0,
                        "error": str(e),
                        "timestamp": request_end
                    })

        # Start workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(config.concurrent_users)]
            concurrent.futures.wait(futures, timeout=config.duration_seconds + 10)

        # Analyze results
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        failed_requests = total_requests - successful_requests

        durations = [r["duration_ms"] for r in results if r["success"]]

        if durations:
            avg_response_time = np.mean(durations)
            p95_response_time = np.percentile(durations, 95)
            p99_response_time = np.percentile(durations, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0

        requests_per_second = total_requests / config.duration_seconds if config.duration_seconds > 0 else 0

        load_test_results = {
            "config": asdict(config),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / total_requests) * 100 if total_requests > 0 else 0,
            "requests_per_second": requests_per_second,
            "avg_response_time_ms": avg_response_time,
            "p95_response_time_ms": p95_response_time,
            "p99_response_time_ms": p99_response_time,
            "duration_seconds": config.duration_seconds,
            "completed_at": datetime.now()
        }

        logger.info(f"Load test completed: {successful_requests}/{total_requests} successful ({requests_per_second:.1f} RPS)")

        return load_test_results

    def run_chaos_test(
        self,
        target_function: Callable,
        config: ChaosTestConfig,
        test_duration_seconds: int = 300
    ) -> dict[str, Any]:
        """Run chaos test with fault injection.
        
        Args:
            target_function: Function to test
            config: Chaos test configuration
            test_duration_seconds: Duration of chaos test
            
        Returns:
            Chaos test results
        """
        logger.info(f"Starting chaos test for {test_duration_seconds}s")

        results = []
        chaos_events = []
        start_time = time.time()
        end_time = start_time + test_duration_seconds

        def inject_chaos():
            """Inject chaos failures."""
            while time.time() < end_time:
                if random.random() < config.failure_probability:
                    failure_type = random.choice(config.failure_types)

                    chaos_event = {
                        "type": failure_type,
                        "start_time": time.time(),
                        "duration": config.failure_duration_seconds
                    }

                    logger.info(f"Injecting chaos: {failure_type}")

                    if failure_type == "memory_pressure":
                        self._inject_memory_pressure(config.failure_duration_seconds)
                    elif failure_type == "cpu_spike":
                        self._inject_cpu_spike(config.failure_duration_seconds)
                    elif failure_type == "network_delay":
                        self._inject_network_delay(config.failure_duration_seconds)

                    chaos_events.append(chaos_event)

                time.sleep(10)  # Check every 10 seconds

        def run_normal_operations():
            """Run normal operations during chaos."""
            while time.time() < end_time:
                try:
                    operation_start = time.time()
                    result = target_function(text="This is a chaos test input")
                    operation_end = time.time()

                    results.append({
                        "success": True,
                        "duration_ms": (operation_end - operation_start) * 1000,
                        "timestamp": operation_start
                    })

                except Exception as e:
                    operation_end = time.time()
                    results.append({
                        "success": False,
                        "duration_ms": (operation_end - operation_start) * 1000 if 'operation_start' in locals() else 0,
                        "error": str(e),
                        "timestamp": operation_end
                    })

                time.sleep(1)  # Normal operation frequency

        # Run chaos test with concurrent chaos injection and normal operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            chaos_future = executor.submit(inject_chaos)
            operations_future = executor.submit(run_normal_operations)

            concurrent.futures.wait([chaos_future, operations_future], timeout=test_duration_seconds + 30)

        # Analyze results
        total_operations = len(results)
        successful_operations = sum(1 for r in results if r["success"])

        chaos_test_results = {
            "config": asdict(config),
            "duration_seconds": test_duration_seconds,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": (successful_operations / total_operations) * 100 if total_operations > 0 else 0,
            "chaos_events": len(chaos_events),
            "chaos_types": list(set(event["type"] for event in chaos_events)),
            "completed_at": datetime.now()
        }

        logger.info(f"Chaos test completed: {len(chaos_events)} chaos events, {successful_operations}/{total_operations} operations succeeded")

        return chaos_test_results

    def _inject_memory_pressure(self, duration_seconds: int) -> None:
        """Inject memory pressure."""
        try:
            # Allocate memory to create pressure
            memory_hog = []
            chunk_size = 10 * 1024 * 1024  # 10MB chunks

            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                try:
                    memory_hog.append(b'x' * chunk_size)
                    time.sleep(0.1)
                except MemoryError:
                    break

            # Clean up
            del memory_hog

        except Exception as e:
            logger.error(f"Error injecting memory pressure: {e}")

    def _inject_cpu_spike(self, duration_seconds: int) -> None:
        """Inject CPU spike."""
        try:
            def cpu_burner():
                end_time = time.time() + duration_seconds
                while time.time() < end_time:
                    # Busy loop to consume CPU
                    for _ in range(1000000):
                        pass

            # Use multiple threads to maximize CPU usage
            cpu_count = multiprocessing.cpu_count()
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
                futures = [executor.submit(cpu_burner) for _ in range(cpu_count)]
                concurrent.futures.wait(futures, timeout=duration_seconds + 5)

        except Exception as e:
            logger.error(f"Error injecting CPU spike: {e}")

    def _inject_network_delay(self, duration_seconds: int) -> None:
        """Inject network delay (simulated)."""
        # This is a placeholder - in a real implementation, you might use
        # network simulation tools or modify network behavior
        logger.info(f"Simulating network delay for {duration_seconds}s")
        time.sleep(min(duration_seconds, 5))  # Simulate some delay

    def get_test_summary(self, test_type: str | None = None) -> dict[str, Any]:
        """Get summary of test results.
        
        Args:
            test_type: Filter by test type
            
        Returns:
            Test summary
        """
        with self._lock:
            results = self.test_results

            if test_type:
                # Filter by test type (approximate matching)
                results = [r for r in results if test_type in r.test_id.lower()]

            if not results:
                return {"message": "No test results available"}

            total_tests = len(results)
            successful_tests = sum(1 for r in results if r.success)

            return {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests / total_tests) * 100,
                "avg_duration_ms": np.mean([r.duration_ms for r in results]),
                "test_types": list(set(r.test_id.split('_')[0] for r in results if '_' in r.test_id)),
                "recent_failures": [
                    {"test_id": r.test_id, "error": r.error}
                    for r in results[-10:] if not r.success
                ]
            }


# Global instances
_schema_validator: SchemaValidator | None = None
_performance_profiler: PerformanceProfiler | None = None
_test_framework: TestFramework | None = None


def get_schema_validator() -> SchemaValidator:
    """Get global schema validator.
    
    Returns:
        SchemaValidator instance
    """
    global _schema_validator
    if _schema_validator is None:
        _schema_validator = SchemaValidator()
    return _schema_validator


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler.
    
    Returns:
        PerformanceProfiler instance
    """
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler


def get_test_framework() -> TestFramework:
    """Get global test framework.
    
    Returns:
        TestFramework instance
    """
    global _test_framework
    if _test_framework is None:
        _test_framework = TestFramework()
    return _test_framework


def validate_schema(schema_name: str):
    """Decorator to validate function inputs/outputs against schema.
    
    Args:
        schema_name: Name of the schema to validate against
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            validator = get_schema_validator()

            # Validate inputs (approximate - would need more sophisticated mapping)
            if kwargs:
                input_result = validator.validate(kwargs, f"{schema_name}_input")
                if not input_result.valid:
                    raise ValidationError(
                        f"Input validation failed: {'; '.join(input_result.errors)}"
                    )

            # Execute function
            result = func(*args, **kwargs)

            # Validate outputs if result is a dictionary
            if isinstance(result, dict):
                output_result = validator.validate(result, f"{schema_name}_output")
                if not output_result.valid:
                    logger.warning(f"Output validation failed: {'; '.join(output_result.errors)}")

            return result

        return wrapper
    return decorator


def profile_performance(include_memory: bool = True):
    """Decorator to profile function performance.
    
    Args:
        include_memory: Whether to include memory profiling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            operation_name = f"{func.__module__}.{func.__name__}"

            with profiler.profile(operation_name, include_memory):
                return func(*args, **kwargs)

        return wrapper
    return decorator
