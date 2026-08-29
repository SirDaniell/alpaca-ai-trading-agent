"""
Credential Encryption Module
Provides encryption/decryption for sensitive data using Fernet symmetric encryption
"""

from cryptography.fernet import Fernet
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CredentialEncryption:
    """Encrypt/decrypt sensitive credentials using Fernet"""
    
    def __init__(self):
        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            raise ValueError("ENCRYPTION_KEY not set in environment variables")
        
        try:
            self.cipher = Fernet(key.encode())
            logger.info("Encryption service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise ValueError(f"Invalid ENCRYPTION_KEY format: {e}")
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt plaintext data
        
        Args:
            data: Plaintext string to encrypt
            
        Returns:
            Base64-encoded encrypted string
        """
        try:
            encrypted_bytes = self.cipher.encrypt(data.encode())
            return encrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError(f"Failed to encrypt data: {e}")
    
    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt encrypted data
        
        Args:
            encrypted: Base64-encoded encrypted string
            
        Returns:
            Decrypted plaintext string
        """
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError(f"Failed to decrypt data: {e}")
    
    def encrypt_dict(self, data: dict) -> str:
        """
        Encrypt a dictionary by converting to JSON
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Encrypted JSON string
        """
        import json
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted: str) -> dict:
        """
        Decrypt an encrypted JSON string back to dictionary
        
        Args:
            encrypted: Encrypted JSON string
            
        Returns:
            Decrypted dictionary
        """
        import json
        json_str = self.decrypt(encrypted)
        return json.loads(json_str)


# Global instance
_encryption = None


def get_encryption() -> CredentialEncryption:
    """
    Get encryption service singleton
    
    Returns:
        CredentialEncryption instance
    """
    global _encryption
    if _encryption is None:
        _encryption = CredentialEncryption()
    return _encryption
