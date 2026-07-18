from __future__ import annotations

from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
import re


MAX_EML_SIZE_BYTES = 15 * 1024 * 1024
SUPPORTED_EML_EXTENSIONS = {".eml"}
SUPPORTED_EML_CONTENT_TYPES = {
    "message/rfc822",
    "application/octet-stream",
    "text/plain",
}

EXECUTABLE_EXTENSIONS = {
    ".exe",
    ".scr",
    ".com",
    ".bat",
    ".cmd",
    ".ps1",
    ".vbs",
    ".vbe",
    ".js",
    ".jse",
    ".wsf",
    ".wsh",
    ".hta",
    ".msi",
    ".msp",
    ".dll",
    ".jar",
    ".lnk",
    ".iso",
    ".img",
}

MACRO_EXTENSIONS = {".docm", ".xlsm", ".pptm", ".xltm", ".dotm"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"}


class _HTMLTextExtractor(HTMLParser):
    """Convert a basic HTML email body into readable plain text."""

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "p",
        "section",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        if lower_tag in {"script", "style"}:
            self._ignored_depth += 1
        elif lower_tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag in {"script", "style"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
        elif lower_tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = unescape("".join(self._parts))
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()


class EMLService:
    @staticmethod
    def _decode_header_value(value: Any) -> str:
        if value is None:
            return ""

        try:
            return str(make_header(decode_header(str(value)))).strip()
        except Exception:
            return str(value).strip()

    @staticmethod
    def _normalise_text(text: str) -> str:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _html_to_text(html_content: str) -> str:
        parser = _HTMLTextExtractor()
        try:
            parser.feed(html_content)
            parser.close()
            return parser.get_text()
        except Exception:
            without_tags = re.sub(r"<[^>]+>", " ", html_content)
            return re.sub(r"\s+", " ", unescape(without_tags)).strip()

    @staticmethod
    def _address_list(header_values: list[str]) -> list[dict[str, str]]:
        addresses: list[dict[str, str]] = []
        for display_name, address in getaddresses(header_values):
            clean_address = address.strip().lower()
            if clean_address:
                addresses.append(
                    {
                        "display_name": display_name.strip(),
                        "address": clean_address,
                    }
                )
        return addresses

    @staticmethod
    def _single_address(header_value: str) -> dict[str, str]:
        display_name, address = parseaddr(header_value or "")
        return {
            "display_name": display_name.strip(),
            "address": address.strip().lower(),
        }

    @staticmethod
    def _domain_from_address(address: str) -> str:
        if "@" not in address:
            return ""
        return address.rsplit("@", 1)[1].strip().lower().strip("<>[]()")

    @staticmethod
    def _message_id_domain(message_id: str) -> str:
        match = re.search(r"@([^>\s]+)", message_id or "")
        if not match:
            return ""
        return match.group(1).strip().lower().strip("<>[]()")

    @staticmethod
    def _extract_authentication_result(
        authentication_text: str,
        mechanism: str,
    ) -> str:
        pattern = re.compile(
            rf"(?i)\b{re.escape(mechanism)}\s*=\s*"
            r"(pass|fail|softfail|neutral|none|temperror|permerror|policy)"
        )
        match = pattern.search(authentication_text or "")
        return match.group(1).lower() if match else "unknown"

    @classmethod
    def _extract_body(cls, message: Any) -> tuple[str, str, bool]:
        plain_parts: list[str] = []
        html_parts: list[str] = []

        parts = message.walk() if message.is_multipart() else [message]

        for part in parts:
            if part.is_multipart():
                continue

            filename = part.get_filename()
            disposition = (part.get_content_disposition() or "").lower()
            if filename or disposition == "attachment":
                continue

            content_type = part.get_content_type().lower()
            if content_type not in {"text/plain", "text/html"}:
                continue

            try:
                content = part.get_content()
            except Exception:
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                content = payload.decode(charset, errors="replace")

            if not isinstance(content, str):
                content = str(content)

            if content_type == "text/plain":
                normalised = cls._normalise_text(content)
                if normalised:
                    plain_parts.append(normalised)
            else:
                html_text = cls._html_to_text(content)
                if html_text:
                    html_parts.append(html_text)

        if plain_parts:
            return "\n\n".join(plain_parts).strip(), "text/plain", bool(html_parts)

        if html_parts:
            return "\n\n".join(html_parts).strip(), "text/html converted to text", True

        return "", "unavailable", False

    @staticmethod
    def _is_double_extension(filename: str) -> bool:
        lower_name = filename.lower()
        suspicious_second_extensions = {
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".jpg",
            ".jpeg",
            ".png",
            ".txt",
        }
        final_suffix = Path(lower_name).suffix
        stem_suffix = Path(Path(lower_name).stem).suffix
        return bool(final_suffix and stem_suffix in suspicious_second_extensions)

    @classmethod
    def _extract_attachments(cls, message: Any) -> list[dict[str, Any]]:
        attachments: list[dict[str, Any]] = []

        for part in message.walk():
            filename = cls._decode_header_value(part.get_filename())
            disposition = (part.get_content_disposition() or "").lower()

            if not filename and disposition != "attachment":
                continue

            payload = part.get_payload(decode=True) or b""
            extension = Path(filename).suffix.lower() if filename else ""
            warnings: list[str] = []
            risk = "Low"

            if extension in EXECUTABLE_EXTENSIONS:
                warnings.append("Executable or script attachment type detected.")
                risk = "Critical"
            elif extension in MACRO_EXTENSIONS:
                warnings.append("Macro-enabled Microsoft Office attachment detected.")
                risk = "High"
            elif extension in ARCHIVE_EXTENSIONS:
                warnings.append("Archive attachment detected; contents were not opened.")
                risk = "Medium"

            if filename and cls._is_double_extension(filename):
                warnings.append("Possible double-extension filename detected.")
                risk = "Critical" if risk != "Critical" else risk

            attachments.append(
                {
                    "filename": filename or "unnamed_attachment",
                    "content_type": part.get_content_type(),
                    "size_bytes": len(payload),
                    "extension": extension,
                    "risk": risk,
                    "warnings": warnings,
                }
            )

        return attachments

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 6:
            return "Critical"
        if score >= 4:
            return "High"
        if score >= 2:
            return "Medium"
        return "Low"

    def parse_eml(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        if not file_bytes:
            raise ValueError("The uploaded .eml file is empty.")

        if len(file_bytes) > MAX_EML_SIZE_BYTES:
            raise ValueError("The .eml file exceeds the 15 MB upload limit.")

        extension = Path(filename or "").suffix.lower()
        if extension not in SUPPORTED_EML_EXTENSIONS:
            raise ValueError("Only original .eml email files are supported.")

        try:
            message = BytesParser(policy=policy.default).parsebytes(file_bytes)
        except Exception as error:
            raise ValueError("The uploaded file could not be parsed as an email.") from error

        subject = self._decode_header_value(message.get("Subject"))
        from_header = self._decode_header_value(message.get("From"))
        reply_to_header = self._decode_header_value(message.get("Reply-To"))
        return_path_header = self._decode_header_value(message.get("Return-Path"))
        message_id = self._decode_header_value(message.get("Message-ID"))
        date_header = self._decode_header_value(message.get("Date"))

        from_address = self._single_address(from_header)
        reply_to_address = self._single_address(reply_to_header)
        return_path_address = self._single_address(return_path_header)

        to_addresses = self._address_list(
            [self._decode_header_value(value) for value in message.get_all("To", [])]
        )
        cc_addresses = self._address_list(
            [self._decode_header_value(value) for value in message.get_all("Cc", [])]
        )

        from_domain = self._domain_from_address(from_address["address"])
        reply_to_domain = self._domain_from_address(reply_to_address["address"])
        return_path_domain = self._domain_from_address(return_path_address["address"])
        message_id_domain = self._message_id_domain(message_id)

        from_replyto_mismatch = bool(
            from_domain and reply_to_domain and from_domain != reply_to_domain
        )
        from_returnpath_mismatch = bool(
            from_domain and return_path_domain and from_domain != return_path_domain
        )
        message_id_domain_mismatch = bool(
            from_domain and message_id_domain and from_domain != message_id_domain
        )

        authentication_headers = [
            self._decode_header_value(value)
            for value in message.get_all("Authentication-Results", [])
        ]
        authentication_text = "\n".join(authentication_headers)

        spf_result = self._extract_authentication_result(authentication_text, "spf")
        dkim_result = self._extract_authentication_result(authentication_text, "dkim")
        dmarc_result = self._extract_authentication_result(authentication_text, "dmarc")

        received_spf = "\n".join(
            self._decode_header_value(value)
            for value in message.get_all("Received-SPF", [])
        )
        if spf_result == "unknown" and received_spf:
            spf_match = re.search(
                r"(?i)\b(pass|fail|softfail|neutral|none|temperror|permerror)\b",
                received_spf,
            )
            if spf_match:
                spf_result = spf_match.group(1).lower()

        body, body_source, has_html_body = self._extract_body(message)
        attachments = self._extract_attachments(message)

        warnings: list[str] = []
        risk_score = 0

        if from_replyto_mismatch:
            warnings.append("The From and Reply-To domains do not match.")
            risk_score += 1

        if from_returnpath_mismatch:
            warnings.append("The From and Return-Path domains do not match.")
            risk_score += 1

        if message_id_domain_mismatch:
            warnings.append("The Message-ID domain does not match the From domain.")
            risk_score += 1

        if spf_result in {"fail", "softfail", "permerror"}:
            warnings.append(f"SPF authentication result is {spf_result}.")
            risk_score += 2

        if dkim_result in {"fail", "permerror"}:
            warnings.append(f"DKIM authentication result is {dkim_result}.")
            risk_score += 2

        if dmarc_result in {"fail", "permerror"}:
            warnings.append(f"DMARC authentication result is {dmarc_result}.")
            risk_score += 2

        if not authentication_headers and not received_spf:
            warnings.append(
                "No SPF, DKIM, or DMARC result was found in the visible headers."
            )

        for attachment in attachments:
            if attachment["risk"] == "Critical":
                risk_score += 3
            elif attachment["risk"] == "High":
                risk_score += 2
            elif attachment["risk"] == "Medium":
                risk_score += 1

        received_count = len(message.get_all("Received", []))

        return {
            "filename": filename,
            "content_type": content_type,
            "extraction_method": "Python email parser",
            "subject": subject,
            "body": body,
            "body_source": body_source,
            "has_html_body": has_html_body,
            "headers": {
                "from": from_header,
                "from_address": from_address["address"],
                "from_display_name": from_address["display_name"],
                "reply_to": reply_to_header,
                "reply_to_address": reply_to_address["address"],
                "return_path": return_path_header,
                "return_path_address": return_path_address["address"],
                "to": to_addresses,
                "cc": cc_addresses,
                "date": date_header,
                "message_id": message_id,
                "received_header_count": received_count,
            },
            "domains": {
                "from_domain": from_domain,
                "reply_to_domain": reply_to_domain,
                "return_path_domain": return_path_domain,
                "message_id_domain": message_id_domain,
            },
            "authentication": {
                "spf": spf_result,
                "dkim": dkim_result,
                "dmarc": dmarc_result,
                "authentication_results": authentication_headers,
                "received_spf": received_spf,
            },
            "security_features": {
                "from_replyto_mismatch": int(from_replyto_mismatch),
                "from_returnpath_mismatch": int(from_returnpath_mismatch),
                "message_id_domain_mismatch": int(message_id_domain_mismatch),
                "attachment_count": len(attachments),
                "has_executable_attachment": int(
                    any(item["extension"] in EXECUTABLE_EXTENSIONS for item in attachments)
                ),
                "has_macro_attachment": int(
                    any(item["extension"] in MACRO_EXTENSIONS for item in attachments)
                ),
            },
            "attachments": attachments,
            "header_risk_score": risk_score,
            "header_risk": self._risk_level(risk_score),
            "warnings": warnings,
        }


eml_service = EMLService()
