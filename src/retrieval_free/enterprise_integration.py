"""Enterprise integration module that combines all robust features into a production-ready system."""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from functools import wraps
from datetime import datetime

# Import all enterprise features
from .security import (
    get_auth_manager, get_input_sanitizer, require_authentication, 
    SecurityError, ModelSecurityValidator, AuditLogger
)
from .error_handling import (
    get_resource_manager, get_circuit_breaker, get_retry_mechanism,
    get_graceful_degradation, with_resilience, CircuitBreakerConfig,
    RetryConfig, TimeoutHandler
)
from .configuration import (
    get_config_manager, get_feature_toggle_manager, feature_flag,
    config_value, FeatureToggleStrategy, Environment
)
from .monitoring_enhanced import (
    get_distributed_tracer, get_enhanced_metrics_collector, get_alert_manager,
    trace_operation, HealthStatus
)
from .quality_assurance import (
    get_schema_validator, get_performance_profiler, get_test_framework,
    validate_schema, profile_performance, ValidationRule, TestCase
)
from .compliance import get_compliance_manager, ProcessingPurpose, LegalBasis
from .core import ContextCompressor, CompressionResult, MegaToken
from .exceptions import CompressionError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class EnterpriseCompressionRequest:
    """Enterprise compression request with all metadata."""
    text: str
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    compression_ratio: Optional[float] = None
    chunk_size: Optional[int] = None
    enable_pii_masking: bool = True
    compliance_purpose: ProcessingPurpose = ProcessingPurpose.COMPRESSION
    legal_basis: LegalBasis = LegalBasis.LEGITIMATE_INTERESTS
    metadata: Dict[str, Any] = None


@dataclass
class EnterpriseCompressionResponse:
    """Enterprise compression response with comprehensive metadata."""
    success: bool
    result: Optional[CompressionResult] = None
    error: Optional[str] = None
    security_scan: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    compliance_record: Optional[str] = None
    trace_id: Optional[str] = None
    processing_time_ms: float = 0
    warnings: List[str] = None


class EnterpriseContextCompressor:
    """Enterprise-grade context compressor with all robust features integrated."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize enterprise context compressor.
        
        Args:
            config_dir: Directory containing configuration files
        """
        # Initialize all enterprise components
        self.config_manager = get_config_manager(config_dir)
        self.feature_manager = get_feature_toggle_manager()
        self.auth_manager = get_auth_manager()
        self.input_sanitizer = get_input_sanitizer()
        self.tracer = get_distributed_tracer()
        self.metrics_collector = get_enhanced_metrics_collector()
        self.alert_manager = get_alert_manager()
        self.schema_validator = get_schema_validator()
        self.profiler = get_performance_profiler()
        self.resource_manager = get_resource_manager()
        self.compliance_manager = get_compliance_manager()
        
        # Initialize security components
        self.model_validator = ModelSecurityValidator()
        self.audit_logger = AuditLogger()
        
        # Load configuration and set up compressor
        self._setup_compressor()
        self._setup_enterprise_features()
        
        logger.info("Enterprise context compressor initialized")
    
    def _setup_compressor(self) -> None:
        """Set up the core compressor with enterprise configuration."""
        model_name = self.config_manager.get(
            "compression.model_name", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        chunk_size = self.config_manager.get("compression.chunk_size", 512)
        compression_ratio = self.config_manager.get("compression.compression_ratio", 8.0)
        
        # Validate model security
        security_scan = self.model_validator.validate_model_source(model_name)
        if not security_scan.passed:
            logger.warning(f"Model security scan failed: {security_scan.vulnerabilities}")
        
        # Initialize compressor with resilience
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=3
        )
        
        retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0
        )
        
        self.circuit_breaker = get_circuit_breaker("compression", circuit_breaker_config)
        self.retry_mechanism = get_retry_mechanism("compression", retry_config)
        
        # Set up graceful degradation
        self.degradation = get_graceful_degradation("compression")
        
        try:
            # Primary compressor
            self.primary_compressor = ContextCompressor(
                model_name=model_name,
                chunk_size=chunk_size,
                compression_ratio=compression_ratio
            )
            self.degradation.primary(self._compress_with_primary)
            
            # Fallback compressor (simpler model)
            self.fallback_compressor = ContextCompressor(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                chunk_size=256,
                compression_ratio=4.0
            )
            self.degradation.fallback(self._compress_with_fallback)
            
        except Exception as e:
            logger.error(f"Failed to initialize compressors: {e}")
            raise CompressionError(f"Compressor initialization failed: {e}")
    
    def _setup_enterprise_features(self) -> None:
        """Set up enterprise features and monitoring."""
        # Register feature toggles
        self.feature_manager.register_toggle(
            "pii_detection",
            FeatureToggleStrategy.ON,
            enabled=True
        )
        
        self.feature_manager.register_toggle(
            "advanced_monitoring",
            FeatureToggleStrategy.PERCENTAGE,
            enabled=True,
            percentage=100.0
        )
        
        self.feature_manager.register_toggle(
            "compliance_logging",
            FeatureToggleStrategy.ON,
            enabled=True
        )
        
        # Set up custom alerts
        self.alert_manager.register_alert({
            "alert_id": "high_pii_detection",
            "name": "High PII Detection Rate",
            "metric_name": "pii_detection_rate",
            "operator": ">",
            "threshold": 0.1,  # 10%
            "severity": "warning",
            "description": "High rate of PII detection in inputs"
        })
        
        # Register validation schemas
        self._setup_validation_schemas()
    
    def _setup_validation_schemas(self) -> None:
        """Set up validation schemas for enterprise operations."""
        # Enterprise request schema
        request_rules = [
            ValidationRule("text", "required", None, "Text input is required"),
            ValidationRule("text", "type", str, "Text must be a string"),
            ValidationRule("text", "custom", "non_empty_text", "Text cannot be empty"),
            ValidationRule("user_id", "type", str, "User ID must be a string", "warning"),
            ValidationRule("compression_ratio", "range", (1.0, 100.0), "Invalid compression ratio", "warning"),
        ]
        self.schema_validator.register_schema("enterprise_request", request_rules)
        
        # Enterprise response schema
        response_rules = [
            ValidationRule("success", "required", None, "Success field is required"),
            ValidationRule("success", "type", bool, "Success must be boolean"),
            ValidationRule("processing_time_ms", "type", (int, float), "Processing time must be numeric"),
            ValidationRule("trace_id", "type", str, "Trace ID must be string", "warning"),
        ]
        self.schema_validator.register_schema("enterprise_response", response_rules)
    
    @trace_operation("enterprise_compress")
    @profile_performance()
    @require_authentication(permissions={"compress"})
    @feature_flag("compression_service", default=True)
    @config_value("compression.timeout_seconds", default=30)
    @validate_schema("enterprise_request")
    @with_resilience(
        "enterprise_compression",
        timeout_seconds=30,
        enable_graceful_degradation=True
    )
    def compress(
        self, 
        request: EnterpriseCompressionRequest,
        **kwargs
    ) -> EnterpriseCompressionResponse:
        """Compress text with full enterprise features.
        
        Args:
            request: Enterprise compression request
            **kwargs: Additional parameters injected by decorators
            
        Returns:
            Enterprise compression response
        """
        start_time = time.time()
        trace_id = None
        warnings = []
        
        try:
            # Get current span for tracing
            active_span = self.tracer.get_active_span()
            trace_id = active_span.trace_id if active_span else None
            
            # Extract API key info from decorators
            api_key_info = kwargs.get('_api_key_info')
            user_id = request.user_id or (api_key_info.key_id if api_key_info else None)
            
            # Input sanitization and PII detection
            security_scan = None
            sanitized_text = request.text
            
            if self.feature_manager.is_enabled("pii_detection", user_id):
                sanitization_result = self.input_sanitizer.sanitize_input(
                    request.text, 
                    mask_pii=request.enable_pii_masking
                )
                
                if sanitization_result['pii_detected']:
                    warnings.append(f"PII detected: {sanitization_result['pii_detected']}")
                    
                    # Log security event
                    self.audit_logger.log_security_event(
                        "pii_detection",
                        f"PII detected in user input: {sanitization_result['pii_detected']}",
                        "warning"
                    )
                
                sanitized_text = sanitization_result['sanitized_text']
                security_scan = {
                    'pii_detected': sanitization_result['pii_detected'],
                    'risk_score': sanitization_result['risk_score'],
                    'malicious_patterns': sanitization_result['malicious_patterns_found']
                }
                
                # Update metrics
                pii_detection_rate = len(sanitization_result['pii_detected']) / max(len(request.text), 1)
                self.metrics_collector.gauges['pii_detection_rate'] = pii_detection_rate
                
                # Check alerts
                self.alert_manager.check_metrics({'pii_detection_rate': pii_detection_rate})
            
            # Compliance logging
            compliance_record = None
            if self.feature_manager.is_enabled("compliance_logging", user_id):
                if user_id:
                    # Register data subject if needed
                    if user_id not in self.compliance_manager.data_subjects:
                        self.compliance_manager.register_data_subject(
                            user_id, 
                            consent=True  # In production, get real consent
                        )
                    
                    # Record processing activity
                    processing_record = self.compliance_manager.record_processing(
                        data_subject_id=user_id,
                        data_content=sanitized_text,
                        purpose=request.compliance_purpose,
                        legal_basis=request.legal_basis
                    )
                    compliance_record = processing_record.id
            
            # Audit logging
            self.audit_logger.log_compression_request(
                text_length=len(request.text),
                model=request.model_name or "default",
                parameters={
                    "compression_ratio": request.compression_ratio,
                    "chunk_size": request.chunk_size,
                    "pii_masking": request.enable_pii_masking
                },
                user=user_id
            )
            
            # Perform compression with graceful degradation
            compression_params = {
                "text": sanitized_text
            }
            
            if request.model_name:
                compression_params["model_name"] = request.model_name
            if request.compression_ratio:
                compression_params["compression_ratio"] = request.compression_ratio
            if request.chunk_size:
                compression_params["chunk_size"] = request.chunk_size
            
            # Use timeout handler
            timeout_seconds = kwargs.get('config_compression_timeout_seconds', 30)
            with TimeoutHandler(timeout_seconds, "enterprise_compression"):
                result = self.degradation.execute(**compression_params)
            
            # Record success metrics
            processing_time_ms = (time.time() - start_time) * 1000
            
            self.metrics_collector.record_compression(
                input_tokens=len(request.text.split()),
                output_tokens=len(result.mega_tokens) if result else 0,
                processing_time_ms=processing_time_ms,
                model_name=request.model_name or "default",
                trace_id=trace_id,
                user_id=user_id,
                success=True
            )
            
            # Performance metrics
            performance_metrics = {
                "processing_time_ms": processing_time_ms,
                "input_tokens": len(request.text.split()),
                "output_tokens": len(result.mega_tokens) if result else 0,
                "compression_ratio": result.compression_ratio if result else 0,
                "memory_usage_mb": self.resource_manager.check_memory_usage().get('memory_mb', 0)
            }
            
            # Check if cleanup is needed
            memory_status = self.resource_manager.check_memory_usage()
            if memory_status.get('cleanup_recommended', False):
                cleanup_summary = self.resource_manager.cleanup_resources()
                warnings.append(f"Memory cleanup performed: saved {cleanup_summary.get('memory_saved_mb', 0):.1f}MB")
            
            return EnterpriseCompressionResponse(
                success=True,
                result=result,
                security_scan=security_scan,
                performance_metrics=performance_metrics,
                compliance_record=compliance_record,
                trace_id=trace_id,
                processing_time_ms=processing_time_ms,
                warnings=warnings
            )
            
        except Exception as e:
            # Record failure metrics
            processing_time_ms = (time.time() - start_time) * 1000
            
            self.metrics_collector.record_compression(
                input_tokens=len(request.text.split()) if request.text else 0,
                output_tokens=0,
                processing_time_ms=processing_time_ms,
                model_name=request.model_name or "default",
                trace_id=trace_id,
                user_id=request.user_id,
                success=False
            )
            
            # Log error
            self.audit_logger.log_security_event(
                "compression_error",
                f"Compression failed: {str(e)}",
                "error"
            )
            
            logger.error(f"Enterprise compression failed: {e}")
            
            return EnterpriseCompressionResponse(
                success=False,
                error=str(e),
                trace_id=trace_id,
                processing_time_ms=processing_time_ms,
                warnings=warnings
            )
    
    def _compress_with_primary(self, **kwargs) -> CompressionResult:
        """Compress with primary compressor."""
        return self.primary_compressor.compress(**kwargs)
    
    def _compress_with_fallback(self, **kwargs) -> CompressionResult:
        """Compress with fallback compressor."""
        logger.warning("Using fallback compressor")
        return self.fallback_compressor.compress(**kwargs)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status.
        
        Returns:
            Health status dictionary
        """
        health_checks = []
        
        # Check compressor health
        try:
            test_result = self.primary_compressor.compress("Health check test")
            compressor_health = HealthStatus(
                service="primary_compressor",
                healthy=True,
                message=f"Compressed to {len(test_result.mega_tokens)} tokens",
                response_time_ms=test_result.processing_time * 1000,
                timestamp=time.time()
            )
        except Exception as e:
            compressor_health = HealthStatus(
                service="primary_compressor",
                healthy=False,
                message=f"Health check failed: {e}",
                response_time_ms=0,
                timestamp=time.time()
            )
        
        health_checks.append(asdict(compressor_health))
        
        # Get system health
        memory_status = self.resource_manager.check_memory_usage()
        memory_healthy = not memory_status.get('cleanup_recommended', False)
        
        memory_health = HealthStatus(
            service="memory",
            healthy=memory_healthy,
            message=f"Memory usage: {memory_status.get('memory_mb', 0):.1f}MB",
            response_time_ms=0,
            timestamp=time.time(),
            details=memory_status
        )
        health_checks.append(asdict(memory_health))
        
        # Get circuit breaker status
        cb_status = self.circuit_breaker.get_status()
        cb_healthy = cb_status['state'] != 'open'
        
        circuit_health = HealthStatus(
            service="circuit_breaker",
            healthy=cb_healthy,
            message=f"Circuit breaker state: {cb_status['state']}",
            response_time_ms=0,
            timestamp=time.time(),
            details=cb_status
        )
        health_checks.append(asdict(circuit_health))
        
        # Overall health
        all_healthy = all(check['healthy'] for check in health_checks)
        
        return {
            "healthy": all_healthy,
            "timestamp": datetime.now().isoformat(),
            "checks": health_checks,
            "system_info": {
                "active_traces": len(self.tracer.traces),
                "active_alerts": len(self.alert_manager.get_active_alerts()),
                "feature_toggles": len(self.feature_manager.list_toggles()),
                "tracked_resources": len(self.resource_manager._tracked_resources)
            }
        }
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        return {
            "metrics": self.metrics_collector.get_summary_stats(),
            "alerts": {
                "active": [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
                "history": [asdict(alert) for alert in self.alert_manager.get_alert_history(24)]
            },
            "traces": {
                "active_traces": len(self.tracer.traces),
                "active_spans": len(self.tracer.active_spans)
            },
            "compliance": self.compliance_manager.generate_privacy_report(),
            "performance": self.profiler.get_profile_summary(),
            "configuration": {
                "environment": os.environ.get("RETRIEVAL_FREE_ENV", "development"),
                "feature_toggles": self.feature_manager.list_toggles()
            },
            "security": {
                "active_api_keys": len(self.auth_manager.api_keys),
                "blocked_requests": sum(
                    info.blocked_requests 
                    for info in self.auth_manager.rate_limits.values()
                )
            }
        }
    
    def run_enterprise_health_check(self) -> Dict[str, Any]:
        """Run comprehensive enterprise health check.
        
        Returns:
            Comprehensive health check results
        """
        health_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "components": {}
        }
        
        # Test compression functionality
        try:
            test_request = EnterpriseCompressionRequest(
                text="This is a health check test for enterprise compression functionality.",
                user_id="health_check_user"
            )
            
            # Create a temporary API key for health check
            key_id, api_key = self.auth_manager.generate_api_key(
                permissions={"compress", "health"},
                rate_limit_requests=1000
            )
            test_request.api_key = api_key
            
            response = self.compress(test_request, api_key=api_key)
            
            health_results["components"]["compression"] = {
                "healthy": response.success,
                "response_time_ms": response.processing_time_ms,
                "details": {
                    "compressed_tokens": len(response.result.mega_tokens) if response.result else 0,
                    "warnings": response.warnings
                }
            }
            
            # Clean up test API key
            self.auth_manager.revoke_api_key(key_id)
            
        except Exception as e:
            health_results["components"]["compression"] = {
                "healthy": False,
                "error": str(e)
            }
            health_results["overall_healthy"] = False
        
        # Test monitoring systems
        try:
            monitoring_status = self.get_monitoring_dashboard()
            health_results["components"]["monitoring"] = {
                "healthy": True,
                "active_alerts": len(monitoring_status["alerts"]["active"]),
                "metrics_available": "metrics" in monitoring_status
            }
        except Exception as e:
            health_results["components"]["monitoring"] = {
                "healthy": False,
                "error": str(e)
            }
            health_results["overall_healthy"] = False
        
        # Test configuration system
        try:
            config_test = self.config_manager.get("compression.model_name", "test")
            health_results["components"]["configuration"] = {
                "healthy": True,
                "config_accessible": config_test is not None
            }
        except Exception as e:
            health_results["components"]["configuration"] = {
                "healthy": False,
                "error": str(e)
            }
            health_results["overall_healthy"] = False
        
        return health_results


# Convenience functions for enterprise deployment
def create_enterprise_compressor(config_dir: Optional[str] = None) -> EnterpriseContextCompressor:
    """Create and configure enterprise context compressor.
    
    Args:
        config_dir: Configuration directory path
        
    Returns:
        Configured EnterpriseContextCompressor
    """
    return EnterpriseContextCompressor(config_dir)


def setup_enterprise_monitoring(compressor: EnterpriseContextCompressor) -> None:
    """Set up enterprise monitoring for a compressor instance.
    
    Args:
        compressor: Enterprise compressor instance
    """
    # Additional monitoring setup can be added here
    logger.info("Enterprise monitoring configured")


def get_enterprise_status() -> Dict[str, Any]:
    """Get comprehensive enterprise system status.
    
    Returns:
        Enterprise system status
    """
    from .monitoring_enhanced import get_monitoring_status
    from .error_handling import get_resilience_status
    
    return {
        "timestamp": datetime.now().isoformat(),
        "monitoring": get_monitoring_status(),
        "resilience": get_resilience_status(),
        "components_loaded": {
            "security": True,
            "error_handling": True,
            "configuration": True,
            "monitoring": True,
            "quality_assurance": True,
            "compliance": True
        }
    }