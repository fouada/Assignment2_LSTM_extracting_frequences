"""
Security Manager
ISO/IEC 25010 Security Compliance
"""

import hashlib
import secrets
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Security management for ISO/IEC 25010 compliance
    
    Implements:
    - Confidentiality: Data encryption
    - Integrity: Data validation and checksums
    - Accountability: Audit logging
    - Authenticity: Access verification
    """
    
    def __init__(self, audit_log_path: Optional[Path] = None):
        """
        Initialize security manager
        
        Args:
            audit_log_path: Path for audit log file
        """
        self.audit_log_path = audit_log_path or Path('security_audit.log')
        self.encryption_key = None
        self._initialize_audit_log()
        
    def _initialize_audit_log(self):
        """Initialize audit log file"""
        if not self.audit_log_path.exists():
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_log_path.touch()
            self.log_audit_event('SYSTEM', 'Audit log initialized', {})
    
    # ========================
    # Confidentiality
    # ========================
    
    def generate_encryption_key(self) -> bytes:
        """
        Generate encryption key
        
        Returns:
            Encryption key
        """
        self.encryption_key = Fernet.generate_key()
        self.log_audit_event('SECURITY', 'Encryption key generated', {})
        return self.encryption_key
    
    def set_encryption_key(self, key: bytes):
        """
        Set encryption key
        
        Args:
            key: Encryption key
        """
        self.encryption_key = key
        self.log_audit_event('SECURITY', 'Encryption key set', {})
    
    def encrypt_data(self, data: str) -> bytes:
        """
        Encrypt sensitive data
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if self.encryption_key is None:
            self.generate_encryption_key()
        
        fernet = Fernet(self.encryption_key)
        encrypted = fernet.encrypt(data.encode())
        
        self.log_audit_event('SECURITY', 'Data encrypted', {
            'data_length': len(data)
        })
        
        return encrypted
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if self.encryption_key is None:
            raise ValueError("Encryption key not set")
        
        fernet = Fernet(self.encryption_key)
        decrypted = fernet.decrypt(encrypted_data).decode()
        
        self.log_audit_event('SECURITY', 'Data decrypted', {
            'data_length': len(decrypted)
        })
        
        return decrypted
    
    def secure_config_value(self, value: Any) -> Dict[str, Any]:
        """
        Secure sensitive configuration value
        
        Args:
            value: Configuration value
            
        Returns:
            Secured value dictionary
        """
        value_str = json.dumps(value)
        encrypted = self.encrypt_data(value_str)
        
        return {
            'encrypted': True,
            'value': base64.b64encode(encrypted).decode(),
            'timestamp': datetime.now().isoformat()
        }
    
    def retrieve_secure_config(self, secured_value: Dict[str, Any]) -> Any:
        """
        Retrieve secured configuration value
        
        Args:
            secured_value: Secured value dictionary
            
        Returns:
            Original value
        """
        if not secured_value.get('encrypted', False):
            return secured_value.get('value')
        
        encrypted = base64.b64decode(secured_value['value'])
        decrypted_str = self.decrypt_data(encrypted)
        
        return json.loads(decrypted_str)
    
    # ========================
    # Integrity
    # ========================
    
    def compute_checksum(self, data: bytes, algorithm: str = 'sha256') -> str:
        """
        Compute checksum for data integrity
        
        Args:
            data: Data to checksum
            algorithm: Hash algorithm
            
        Returns:
            Checksum string
        """
        if algorithm == 'sha256':
            hash_obj = hashlib.sha256(data)
        elif algorithm == 'sha512':
            hash_obj = hashlib.sha512(data)
        elif algorithm == 'md5':
            hash_obj = hashlib.md5(data)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        checksum = hash_obj.hexdigest()
        
        self.log_audit_event('INTEGRITY', 'Checksum computed', {
            'algorithm': algorithm,
            'data_size': len(data),
            'checksum': checksum[:16] + '...'
        })
        
        return checksum
    
    def verify_checksum(
        self,
        data: bytes,
        expected_checksum: str,
        algorithm: str = 'sha256'
    ) -> bool:
        """
        Verify data integrity with checksum
        
        Args:
            data: Data to verify
            expected_checksum: Expected checksum
            algorithm: Hash algorithm
            
        Returns:
            True if checksum matches
        """
        computed_checksum = self.compute_checksum(data, algorithm)
        is_valid = computed_checksum == expected_checksum
        
        self.log_audit_event('INTEGRITY', 'Checksum verification', {
            'algorithm': algorithm,
            'result': 'PASS' if is_valid else 'FAIL'
        })
        
        return is_valid
    
    def compute_file_checksum(
        self,
        file_path: Path,
        algorithm: str = 'sha256'
    ) -> str:
        """
        Compute checksum for file
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm
            
        Returns:
            Checksum string
        """
        with open(file_path, 'rb') as f:
            data = f.read()
        
        checksum = self.compute_checksum(data, algorithm)
        
        self.log_audit_event('INTEGRITY', 'File checksum computed', {
            'file': str(file_path),
            'algorithm': algorithm,
            'checksum': checksum[:16] + '...'
        })
        
        return checksum
    
    def verify_file_integrity(
        self,
        file_path: Path,
        expected_checksum: str,
        algorithm: str = 'sha256'
    ) -> bool:
        """
        Verify file integrity
        
        Args:
            file_path: Path to file
            expected_checksum: Expected checksum
            algorithm: Hash algorithm
            
        Returns:
            True if file is intact
        """
        computed_checksum = self.compute_file_checksum(file_path, algorithm)
        is_valid = computed_checksum == expected_checksum
        
        self.log_audit_event('INTEGRITY', 'File integrity verification', {
            'file': str(file_path),
            'result': 'PASS' if is_valid else 'FAIL'
        })
        
        if not is_valid:
            logger.warning(f"File integrity check failed: {file_path}")
        
        return is_valid
    
    # ========================
    # Accountability
    # ========================
    
    def log_audit_event(
        self,
        category: str,
        action: str,
        details: Dict[str, Any],
        user: str = 'system'
    ):
        """
        Log audit event
        
        Args:
            category: Event category
            action: Action performed
            details: Event details
            user: User performing action
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'user': user,
            'action': action,
            'details': details
        }
        
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_audit_log(
        self,
        category: Optional[str] = None,
        user: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit log entries
        
        Args:
            category: Filter by category
            user: Filter by user
            limit: Maximum number of entries
            
        Returns:
            List of audit log entries
        """
        entries = []
        
        try:
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Apply filters
                        if category and entry.get('category') != category:
                            continue
                        if user and entry.get('user') != user:
                            continue
                        
                        entries.append(entry)
                        
                        if limit and len(entries) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            logger.warning(f"Audit log not found: {self.audit_log_path}")
        
        return entries
    
    def generate_audit_report(self, output_path: Path):
        """
        Generate audit report
        
        Args:
            output_path: Path for report output
        """
        entries = self.get_audit_log()
        
        # Categorize events
        by_category = {}
        by_user = {}
        
        for entry in entries:
            category = entry.get('category', 'UNKNOWN')
            user = entry.get('user', 'unknown')
            
            by_category[category] = by_category.get(category, 0) + 1
            by_user[user] = by_user.get(user, 0) + 1
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_events': len(entries),
            'by_category': by_category,
            'by_user': by_user,
            'recent_events': entries[-10:] if entries else []
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log_audit_event('REPORTING', 'Audit report generated', {
            'output_path': str(output_path),
            'total_events': len(entries)
        })
    
    # ========================
    # Authenticity
    # ========================
    
    def generate_api_key(self, user_id: str) -> str:
        """
        Generate API key for user
        
        Args:
            user_id: User identifier
            
        Returns:
            API key
        """
        # Generate secure random token
        token = secrets.token_urlsafe(32)
        
        # Create API key with metadata
        api_key = f"{user_id}:{token}"
        
        self.log_audit_event('AUTHENTICATION', 'API key generated', {
            'user_id': user_id
        })
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> tuple[bool, Optional[str]]:
        """
        Validate API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            Tuple of (is_valid, user_id)
        """
        try:
            parts = api_key.split(':', 1)
            if len(parts) != 2:
                self.log_audit_event('AUTHENTICATION', 'Invalid API key format', {})
                return False, None
            
            user_id, token = parts
            
            # In production, verify against stored keys
            # For now, just validate format
            is_valid = len(token) == 43  # URL-safe base64 encoded 32 bytes
            
            self.log_audit_event('AUTHENTICATION', 'API key validation', {
                'user_id': user_id,
                'result': 'PASS' if is_valid else 'FAIL'
            })
            
            return is_valid, user_id if is_valid else None
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return False, None
    
    def log_access_attempt(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool
    ):
        """
        Log access attempt
        
        Args:
            user_id: User identifier
            resource: Resource accessed
            action: Action attempted
            success: Whether access was granted
        """
        self.log_audit_event('ACCESS', f'Access attempt: {action}', {
            'resource': resource,
            'success': success
        }, user=user_id)
    
    # ========================
    # Utility
    # ========================
    
    def sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize data before logging (remove sensitive info)
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        sensitive_keys = ['password', 'token', 'key', 'secret', 'api_key']
        
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '***REDACTED***'
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_log_data(value)
            else:
                sanitized[key] = value
        
        return sanitized

