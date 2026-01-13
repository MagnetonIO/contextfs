"""Client-side end-to-end encryption for ContextFS.

Implements AES-256-GCM encryption with HKDF key derivation.
The encryption key is derived from the API key + salt, meaning
the server never has access to the plaintext content.

Security properties:
- AES-256-GCM provides authenticated encryption
- HKDF ensures proper key derivation from API key
- Random 12-byte nonces prevent replay attacks
- Server stores only ciphertext
"""

import base64
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Constants
NONCE_SIZE = 12  # 96 bits for GCM
KEY_SIZE = 32  # 256 bits for AES-256
INFO = b"contextfs-e2ee-v1"


@dataclass
class EncryptedContent:
    """Represents encrypted content with its nonce."""

    ciphertext: bytes
    nonce: bytes

    def to_base64(self) -> str:
        """Encode as base64 string (nonce || ciphertext).

        Returns:
            Base64-encoded string of nonce concatenated with ciphertext
        """
        combined = self.nonce + self.ciphertext
        return base64.urlsafe_b64encode(combined).decode("ascii")

    @classmethod
    def from_base64(cls, data: str) -> "EncryptedContent":
        """Decode from base64 string.

        Args:
            data: Base64-encoded string

        Returns:
            EncryptedContent instance

        Raises:
            ValueError: If data is too short to contain nonce
        """
        combined = base64.urlsafe_b64decode(data)
        if len(combined) < NONCE_SIZE:
            raise ValueError("Encrypted data too short")
        return cls(
            nonce=combined[:NONCE_SIZE],
            ciphertext=combined[NONCE_SIZE:],
        )


def derive_encryption_key(api_key: str, salt: str) -> bytes:
    """Derive an AES-256 encryption key from API key and salt.

    Uses HKDF-SHA256 to derive a cryptographically secure key from
    the API key and server-provided salt.

    Args:
        api_key: The full API key (ctxfs_...)
        salt: Base64-encoded salt from server

    Returns:
        32-byte encryption key
    """
    # Add padding if missing (token_urlsafe doesn't include padding)
    padded_salt = salt + "=" * (-len(salt) % 4)
    salt_bytes = base64.urlsafe_b64decode(padded_salt)

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=salt_bytes,
        info=INFO,
    )

    return hkdf.derive(api_key.encode("utf-8"))


def derive_encryption_key_base64(api_key: str, salt: str) -> str:
    """Derive encryption key and return as base64.

    This is the format stored in the config file.

    Args:
        api_key: The full API key
        salt: Base64-encoded salt

    Returns:
        Base64-encoded encryption key
    """
    key = derive_encryption_key(api_key, salt)
    return base64.urlsafe_b64encode(key).decode("ascii")


def encrypt_content(content: str, key: bytes) -> str:
    """Encrypt content using AES-256-GCM.

    Args:
        content: Plaintext content to encrypt
        key: 32-byte encryption key

    Returns:
        Base64-encoded encrypted content (nonce || ciphertext)
    """
    if len(key) != KEY_SIZE:
        raise ValueError(f"Key must be {KEY_SIZE} bytes")

    nonce = os.urandom(NONCE_SIZE)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, content.encode("utf-8"), None)

    encrypted = EncryptedContent(ciphertext=ciphertext, nonce=nonce)
    return encrypted.to_base64()


def decrypt_content(encrypted_data: str, key: bytes) -> str:
    """Decrypt content using AES-256-GCM.

    Args:
        encrypted_data: Base64-encoded encrypted content
        key: 32-byte encryption key

    Returns:
        Decrypted plaintext content

    Raises:
        ValueError: If decryption fails (wrong key or tampered data)
    """
    if len(key) != KEY_SIZE:
        raise ValueError(f"Key must be {KEY_SIZE} bytes")

    encrypted = EncryptedContent.from_base64(encrypted_data)
    aesgcm = AESGCM(key)

    try:
        plaintext = aesgcm.decrypt(encrypted.nonce, encrypted.ciphertext, None)
        return plaintext.decode("utf-8")
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}") from e


class ClientCrypto:
    """Client-side encryption manager.

    Handles all encryption/decryption operations using a derived key.
    """

    def __init__(self, encryption_key: bytes | None = None):
        """Initialize the crypto manager.

        Args:
            encryption_key: Pre-derived 32-byte encryption key
        """
        self._key = encryption_key

    @classmethod
    def from_api_key(cls, api_key: str, salt: str) -> "ClientCrypto":
        """Create a ClientCrypto instance from API key and salt.

        Args:
            api_key: The full API key
            salt: Base64-encoded salt from server

        Returns:
            ClientCrypto instance with derived key
        """
        key = derive_encryption_key(api_key, salt)
        return cls(encryption_key=key)

    @classmethod
    def from_base64_key(cls, key_base64: str) -> "ClientCrypto":
        """Create a ClientCrypto instance from a base64-encoded key.

        Args:
            key_base64: Base64-encoded encryption key (from config)

        Returns:
            ClientCrypto instance
        """
        key = base64.urlsafe_b64decode(key_base64)
        return cls(encryption_key=key)

    @property
    def is_configured(self) -> bool:
        """Check if encryption is configured."""
        return self._key is not None

    def encrypt(self, content: str) -> str:
        """Encrypt content.

        Args:
            content: Plaintext to encrypt

        Returns:
            Base64-encoded ciphertext

        Raises:
            RuntimeError: If encryption key not configured
        """
        if self._key is None:
            raise RuntimeError("Encryption key not configured")
        return encrypt_content(content, self._key)

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt content.

        Args:
            encrypted_data: Base64-encoded ciphertext

        Returns:
            Decrypted plaintext

        Raises:
            RuntimeError: If encryption key not configured
            ValueError: If decryption fails
        """
        if self._key is None:
            raise RuntimeError("Encryption key not configured")
        return decrypt_content(encrypted_data, self._key)

    def encrypt_memory(self, memory: dict) -> dict:
        """Encrypt a memory's content field.

        Args:
            memory: Memory dictionary with 'content' field

        Returns:
            Memory dictionary with encrypted content and 'encrypted' flag
        """
        if not self.is_configured:
            return memory

        result = memory.copy()
        if "content" in result and result["content"]:
            result["encrypted_content"] = self.encrypt(result["content"])
            result["content"] = None  # Clear plaintext
            result["encrypted"] = True
        return result

    def decrypt_memory(self, memory: dict) -> dict:
        """Decrypt a memory's content field.

        Args:
            memory: Memory dictionary with 'encrypted_content' field

        Returns:
            Memory dictionary with decrypted content
        """
        if not self.is_configured:
            return memory

        result = memory.copy()
        if result.get("encrypted") and "encrypted_content" in result:
            result["content"] = self.decrypt(result["encrypted_content"])
            del result["encrypted_content"]
            del result["encrypted"]
        return result

    def get_key_base64(self) -> str | None:
        """Get the encryption key as base64 for storage.

        Returns:
            Base64-encoded key, or None if not configured
        """
        if self._key is None:
            return None
        return base64.urlsafe_b64encode(self._key).decode("ascii")
