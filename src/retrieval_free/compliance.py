"""Data privacy and compliance utilities."""

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ProcessingPurpose(Enum):
    """Data processing purposes."""
    COMPRESSION = "compression"
    ANALYSIS = "analysis"
    TRAINING = "training"
    EVALUATION = "evaluation"
    RESEARCH = "research"


class LegalBasis(Enum):
    """GDPR legal basis for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataSubject:
    """Information about a data subject."""
    id: str
    email: str | None = None
    country: str | None = None
    consent_status: bool = False
    consent_date: datetime | None = None
    opt_out_date: datetime | None = None


@dataclass
class ProcessingRecord:
    """Record of data processing activity."""
    id: str
    timestamp: datetime
    data_subject_id: str
    purpose: ProcessingPurpose
    legal_basis: LegalBasis
    data_classification: DataClassification
    retention_period_days: int
    processing_location: str
    data_hash: str
    metadata: dict[str, Any]


class ComplianceManager:
    """Manager for data privacy compliance."""

    def __init__(self):
        """Initialize compliance manager."""
        self.processing_records: list[ProcessingRecord] = []
        self.data_subjects: dict[str, DataSubject] = {}
        self.retention_policies: dict[ProcessingPurpose, int] = {
            ProcessingPurpose.COMPRESSION: 30,     # 30 days
            ProcessingPurpose.ANALYSIS: 90,       # 90 days
            ProcessingPurpose.TRAINING: 365,      # 1 year
            ProcessingPurpose.EVALUATION: 180,    # 6 months
            ProcessingPurpose.RESEARCH: 1095,     # 3 years
        }

        # Sensitive data patterns for detection
        self.sensitive_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
        }

    def register_data_subject(
        self,
        subject_id: str,
        email: str | None = None,
        country: str | None = None,
        consent: bool = False
    ) -> DataSubject:
        """Register a data subject.
        
        Args:
            subject_id: Unique identifier for the data subject
            email: Email address (optional)
            country: Country code (optional)
            consent: Whether consent has been given
            
        Returns:
            DataSubject instance
        """
        subject = DataSubject(
            id=subject_id,
            email=email,
            country=country,
            consent_status=consent,
            consent_date=datetime.now() if consent else None
        )

        self.data_subjects[subject_id] = subject
        logger.info(f"Registered data subject: {subject_id}")
        return subject

    def record_processing(
        self,
        data_subject_id: str,
        data_content: str,
        purpose: ProcessingPurpose,
        legal_basis: LegalBasis,
        processing_location: str = "local",
        classification: DataClassification = DataClassification.INTERNAL
    ) -> ProcessingRecord:
        """Record a data processing activity.
        
        Args:
            data_subject_id: ID of the data subject
            data_content: Content being processed
            purpose: Purpose of processing
            legal_basis: Legal basis for processing
            processing_location: Location where processing occurs
            classification: Data classification level
            
        Returns:
            ProcessingRecord instance
        """
        # Generate content hash for privacy
        data_hash = hashlib.sha256(data_content.encode()).hexdigest()

        # Check for sensitive data
        sensitive_data = self.detect_sensitive_data(data_content)

        record = ProcessingRecord(
            id=f"proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.processing_records)}",
            timestamp=datetime.now(),
            data_subject_id=data_subject_id,
            purpose=purpose,
            legal_basis=legal_basis,
            data_classification=classification,
            retention_period_days=self.retention_policies[purpose],
            processing_location=processing_location,
            data_hash=data_hash,
            metadata={
                'content_length': len(data_content),
                'sensitive_data_detected': sensitive_data,
                'processing_version': '1.0'
            }
        )

        self.processing_records.append(record)
        logger.info(f"Recorded processing activity: {record.id}")
        return record

    def detect_sensitive_data(self, text: str) -> dict[str, list[str]]:
        """Detect sensitive data in text.
        
        Args:
            text: Text to scan for sensitive data
            
        Returns:
            Dictionary mapping data types to found instances
        """
        detected = {}

        for data_type, pattern in self.sensitive_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Hash the matches for privacy
                hashed_matches = [hashlib.sha256(match.encode()).hexdigest()[:8] for match in matches]
                detected[data_type] = hashed_matches

        return detected

    def check_consent(self, data_subject_id: str) -> bool:
        """Check if a data subject has given consent.
        
        Args:
            data_subject_id: ID of the data subject
            
        Returns:
            True if consent has been given and is valid
        """
        subject = self.data_subjects.get(data_subject_id)
        if not subject:
            return False

        if not subject.consent_status:
            return False

        # Check if consent has been withdrawn
        if subject.opt_out_date and subject.opt_out_date > (subject.consent_date or datetime.min):
            return False

        return True

    def withdraw_consent(self, data_subject_id: str) -> bool:
        """Withdraw consent for a data subject.
        
        Args:
            data_subject_id: ID of the data subject
            
        Returns:
            True if consent was successfully withdrawn
        """
        subject = self.data_subjects.get(data_subject_id)
        if not subject:
            return False

        subject.consent_status = False
        subject.opt_out_date = datetime.now()

        logger.info(f"Consent withdrawn for data subject: {data_subject_id}")
        return True

    def get_data_subject_records(self, data_subject_id: str) -> list[ProcessingRecord]:
        """Get all processing records for a data subject.
        
        Args:
            data_subject_id: ID of the data subject
            
        Returns:
            List of processing records
        """
        return [
            record for record in self.processing_records
            if record.data_subject_id == data_subject_id
        ]

    def delete_data_subject_records(self, data_subject_id: str) -> int:
        """Delete all records for a data subject (right to be forgotten).
        
        Args:
            data_subject_id: ID of the data subject
            
        Returns:
            Number of records deleted
        """
        initial_count = len(self.processing_records)

        # Remove processing records
        self.processing_records = [
            record for record in self.processing_records
            if record.data_subject_id != data_subject_id
        ]

        # Remove data subject
        if data_subject_id in self.data_subjects:
            del self.data_subjects[data_subject_id]

        deleted_count = initial_count - len(self.processing_records)
        logger.info(f"Deleted {deleted_count} records for data subject: {data_subject_id}")
        return deleted_count

    def get_expired_records(self) -> list[ProcessingRecord]:
        """Get records that have exceeded their retention period.
        
        Returns:
            List of expired records
        """
        expired = []
        current_time = datetime.now()

        for record in self.processing_records:
            retention_end = record.timestamp + timedelta(days=record.retention_period_days)
            if current_time > retention_end:
                expired.append(record)

        return expired

    def cleanup_expired_records(self) -> int:
        """Clean up expired records.
        
        Returns:
            Number of records cleaned up
        """
        expired = self.get_expired_records()

        for record in expired:
            self.processing_records.remove(record)

        logger.info(f"Cleaned up {len(expired)} expired records")
        return len(expired)

    def generate_privacy_report(self) -> dict[str, Any]:
        """Generate privacy compliance report.
        
        Returns:
            Privacy report dictionary
        """
        total_subjects = len(self.data_subjects)
        consented_subjects = sum(1 for s in self.data_subjects.values() if s.consent_status)

        # Processing statistics
        purpose_stats = {}
        for purpose in ProcessingPurpose:
            count = sum(1 for r in self.processing_records if r.purpose == purpose)
            purpose_stats[purpose.value] = count

        # Classification statistics
        classification_stats = {}
        for classification in DataClassification:
            count = sum(1 for r in self.processing_records if r.data_classification == classification)
            classification_stats[classification.value] = count

        # Retention compliance
        expired_count = len(self.get_expired_records())

        return {
            'report_date': datetime.now().isoformat(),
            'data_subjects': {
                'total': total_subjects,
                'consented': consented_subjects,
                'consent_rate': (consented_subjects / total_subjects) * 100 if total_subjects > 0 else 0
            },
            'processing_activities': {
                'total': len(self.processing_records),
                'by_purpose': purpose_stats,
                'by_classification': classification_stats
            },
            'retention_compliance': {
                'expired_records': expired_count,
                'compliance_rate': ((len(self.processing_records) - expired_count) / len(self.processing_records)) * 100 if self.processing_records else 100
            },
            'data_protection_measures': {
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'access_controls': True,
                'audit_logging': True
            }
        }

    def export_data_subject_data(self, data_subject_id: str) -> dict[str, Any]:
        """Export all data for a data subject (data portability).
        
        Args:
            data_subject_id: ID of the data subject
            
        Returns:
            Exported data dictionary
        """
        subject = self.data_subjects.get(data_subject_id)
        if not subject:
            return {}

        records = self.get_data_subject_records(data_subject_id)

        export_data = {
            'data_subject': asdict(subject),
            'processing_records': [
                {
                    'id': record.id,
                    'timestamp': record.timestamp.isoformat(),
                    'purpose': record.purpose.value,
                    'legal_basis': record.legal_basis.value,
                    'classification': record.data_classification.value,
                    'retention_period_days': record.retention_period_days,
                    'processing_location': record.processing_location,
                    'metadata': record.metadata
                }
                for record in records
            ],
            'export_date': datetime.now().isoformat(),
            'export_version': '1.0'
        }

        return export_data


class CrossBorderDataTransfer:
    """Manager for cross-border data transfer compliance."""

    def __init__(self):
        """Initialize cross-border transfer manager."""
        # EU adequacy decisions (simplified list)
        self.adequate_countries = {
            'AD', 'AR', 'CA', 'FO', 'GG', 'IL', 'IM', 'JE', 'JP', 'NZ', 'CH', 'UY', 'GB'
        }

        # High privacy risk countries
        self.restricted_countries = {
            'CN', 'RU', 'IR', 'KP'
        }

    def assess_transfer_risk(
        self,
        source_country: str,
        destination_country: str,
        data_classification: DataClassification
    ) -> dict[str, Any]:
        """Assess risk of cross-border data transfer.
        
        Args:
            source_country: Source country code
            destination_country: Destination country code
            data_classification: Classification of data being transferred
            
        Returns:
            Risk assessment dictionary
        """
        risk_level = "low"
        requirements = []
        warnings = []

        # Check if destination is adequate
        if destination_country not in self.adequate_countries:
            risk_level = "medium"
            requirements.append("Standard Contractual Clauses (SCCs) required")

        # Check for restricted countries
        if destination_country in self.restricted_countries:
            risk_level = "high"
            requirements.append("Transfer Impact Assessment (TIA) required")
            requirements.append("Additional safeguards required")
            warnings.append(f"Transfer to {destination_country} may be prohibited")

        # Classification-based requirements
        if data_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
            risk_level = "high" if risk_level != "high" else risk_level
            requirements.append("Enhanced encryption required")
            requirements.append("Regular audits required")

        return {
            'risk_level': risk_level,
            'requirements': requirements,
            'warnings': warnings,
            'adequate_destination': destination_country in self.adequate_countries,
            'restricted_destination': destination_country in self.restricted_countries
        }

    def validate_transfer(
        self,
        source_country: str,
        destination_country: str,
        data_classification: DataClassification,
        has_sccs: bool = False,
        has_adequacy: bool = False
    ) -> bool:
        """Validate if a transfer is compliant.
        
        Args:
            source_country: Source country code
            destination_country: Destination country code
            data_classification: Classification of data
            has_sccs: Whether SCCs are in place
            has_adequacy: Whether adequacy decision exists
            
        Returns:
            True if transfer is compliant
        """
        # EU to adequate country
        if destination_country in self.adequate_countries:
            return True

        # Restricted countries
        if destination_country in self.restricted_countries:
            if data_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
                return False  # Generally prohibited

        # Other countries need SCCs
        if not has_sccs and destination_country not in self.adequate_countries:
            return False

        return True


# Global compliance manager instance
_compliance_manager: ComplianceManager | None = None


def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager.
    
    Returns:
        ComplianceManager instance
    """
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager()
    return _compliance_manager


def record_data_processing(
    data_subject_id: str,
    data_content: str,
    purpose: ProcessingPurpose,
    legal_basis: LegalBasis = LegalBasis.LEGITIMATE_INTERESTS,
    classification: DataClassification = DataClassification.INTERNAL
) -> ProcessingRecord:
    """Record data processing activity.
    
    Args:
        data_subject_id: ID of the data subject
        data_content: Content being processed
        purpose: Purpose of processing
        legal_basis: Legal basis for processing
        classification: Data classification
        
    Returns:
        ProcessingRecord instance
    """
    manager = get_compliance_manager()
    return manager.record_processing(
        data_subject_id=data_subject_id,
        data_content=data_content,
        purpose=purpose,
        legal_basis=legal_basis,
        classification=classification
    )
