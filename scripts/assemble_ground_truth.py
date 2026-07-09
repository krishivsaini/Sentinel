"""Assemble data/ground_truth.jsonl from hand-authored answers + verified drafts (LLM-free).

For each seed question: use the hand-authored (reference_answer, [context_chunk_ids]) below if
present; otherwise reuse the verified "ok" Gemini draft (its answer + cited chunks). All context
chunk_ids are resolved to their exact text from the canonical chunk store, so relevant_contexts
are real corpus passages (what context_recall is scored against). Inline [chunk_id] citation
markers are stripped so gold answers are clean prose.

Run:  python scripts/assemble_ground_truth.py
"""

from __future__ import annotations

import json
import re

from sentinel.config import CHUNKS_PATH, DATA_DIR, GROUND_TRUTH_PATH
from sentinel.schema import Chunk, GroundTruthItem

SEED_PATH = DATA_DIR / "gt_seed_questions.jsonl"
DRAFT_PATH = DATA_DIR / "ground_truth.draft.jsonl"
_CITE_MARKER = re.compile(r"\s*\[rfc\d+#\d+\]")

# index -> (reference_answer, [context_chunk_ids]). Hand-authored from the corpus (LLM-free);
# the other indices reuse their verified "ok" draft.
AUTHORED: dict[int, tuple[str, list[str]]] = {
    0: ("In IETF RFCs the key words indicate requirement levels: MUST (equivalently REQUIRED or "
        "SHALL) means an absolute requirement of the specification; SHOULD (RECOMMENDED) means "
        "there may be valid reasons in particular circumstances to ignore it, but the full "
        "implications must be understood and weighed first; and MAY (OPTIONAL) means an item is "
        "truly optional, so interoperable implementations may choose to include it or not.",
        ["rfc2119#0000", "rfc2119#0001"]),
    17: ("A cookie is only returned to a host whose name domain-matches the cookie's domain. A "
         "string domain-matches a domain string when they are identical, or when the domain "
         "string is a suffix of the host, the character in the host immediately before that "
         "suffix is \".\", and the host is a name (not an IP address). For example, a cookie with "
         "Domain \"example.com\" is sent to foo.example.com, but one with Domain "
         "\"bar.example.com\" is not.",
         ["rfc6265#0018", "rfc6265#0013"]),
    19: ("The server completes the WebSocket opening handshake by returning HTTP status code 101 "
         "(Switching Protocols); any status code other than 101 means the handshake has not "
         "completed and normal HTTP semantics still apply.",
         ["rfc6455#0012"]),
    20: ("The Sec-WebSocket-Key request header carries a client-generated nonce. The server proves "
         "it received and understood the opening handshake by including in the Sec-WebSocket-Accept "
         "response header a hash of that nonce combined with a predefined GUID; any other value "
         "must not be interpreted as acceptance of the connection.",
         ["rfc6455#0012"]),
    21: ("A stateless reset lets an endpoint that has lost the state for a connection terminate it. "
         "It uses a stateless reset token specific to a connection ID (issued in a "
         "NEW_CONNECTION_ID frame, or via the stateless_reset_token transport parameter for the "
         "connection ID chosen during the handshake). When an endpoint receives packets it cannot "
         "process, it sends a packet ending in that token so the peer recognizes the connection "
         "cannot continue and tears it down.",
         ["rfc9000#0082"]),
    22: ("QUIC assigns strictly increasing packet numbers that directly encode transmission order, "
         "which simplifies loss detection. A packet is declared lost based on packet number (e.g., "
         "when later-sent packets are acknowledged), and the contents of a lost packet are not "
         "retransmitted as-is: the information is carried again in new frames in a new packet with "
         "a new packet number.",
         ["rfc9002#0005", "rfc9000#0107"]),
    23: ("A connection ID is an identifier used to identify a QUIC connection at an endpoint, "
         "independent of the network 4-tuple and opaque to the peer. Each endpoint selects one or "
         "more connection IDs for its peer to include in packets sent to it. Because the connection "
         "is identified by the connection ID rather than by IP address and port, connections can "
         "migrate to a new network path (e.g., after NAT rebinding) without breaking.",
         ["rfc9000#0010", "rfc9000#0008"]),
    24: ("The Maximum Segment Lifetime (MSL) is the maximum time a TCP segment can remain in the "
         "network before it must be discarded. It matters because TCP relies on old duplicate "
         "segments draining from the network within an MSL so that sequence numbers can be reused "
         "safely without a wandering duplicate corrupting a later connection; the large sequence "
         "space is sized against it. RFC 793 assumed an MSL of two minutes.",
         ["rfc9293#0028", "rfc9293#0071"]),
    25: ("In the three-way handshake the initiating peer sends a SYN carrying its initial sequence "
         "number; the other peer replies with its own SYN that also acknowledges the first SYN (a "
         "SYN,ACK); and the initiator then sends an ACK of the peer's SYN, after which both peers "
         "are ESTABLISHED. Its principal purpose is to synchronize both peers' sequence numbers and "
         "prevent old duplicate connection initiations from causing confusion.",
         ["rfc9293#0032", "rfc9293#0033"]),
    26: ("The receive window, advertised in the window field of each segment, tells the sender the "
         "range of sequence numbers (how much data) the receiver is currently prepared to accept. "
         "It provides flow control so the sender does not overrun the receiver's buffer; when the "
         "receiver advertises a zero window the sender stops sending data (apart from zero-window "
         "probes) until space is advertised again.",
         ["rfc9293#0054", "rfc9293#0056"]),
    27: ("When the server authenticates with a certificate it sends that certificate in the "
         "Certificate handshake message, followed by a CertificateVerify message. In TLS 1.3 either "
         "a PSK or a certificate is used; a certificate-authenticating server always sends the "
         "Certificate and CertificateVerify messages.",
         ["rfc8446#0025"]),
    28: ("The Application-Layer Protocol Negotiation (ALPN) extension lets the client and server "
         "negotiate which application-layer protocol to use over a TLS connection entirely within "
         "the ClientHello/ServerHello exchange, adding no extra round-trips. This lets several "
         "protocols share one port (such as 443) and lets the server select the protocol, enabling "
         "certificate selection or connection routing based on the negotiated protocol.",
         ["rfc7301#0002", "rfc7301#0005"]),
    29: ("To indicate the target server name, the client includes a \"server_name\" extension in "
         "its (extended) ClientHello. The extension_data carries a ServerNameList; for a DNS "
         "hostname the entry uses name_type \"host_name\" with a HostName holding the fully "
         "qualified domain name. This lets a server hosting multiple names select the correct "
         "certificate or virtual host.",
         ["rfc6066#0005"]),
    30: ("The Subject field identifies the entity associated with the public key stored in the "
         "certificate. Within the tbsCertificate it is the subject name bound to that public key, "
         "alongside the issuer name, a validity period, and other associated information.",
         ["rfc5280#0022"]),
    31: ("An OCSP response conveys the status of the requested certificate as of the time in "
         "thisUpdate, optionally with a nextUpdate (and, for a revoked certificate, a "
         "revocationTime). If the responder is operational but cannot return a status, it instead "
         "returns an exception response such as \"tryLater\", \"sigRequired\" (the client must sign "
         "the request), or \"unauthorized\".",
         ["rfc6960#0008"]),
    32: ("In the http-01 challenge the client proves control of a domain by provisioning a file "
         "whose body is the challenge's key authorization at the path "
         "/.well-known/acme-challenge/{token} on the domain. The ACME server validates by building "
         "the URL \"http://{domain}/.well-known/acme-challenge/{token}\", fetching it with an HTTP "
         "GET to TCP port 80, and verifying that the response body matches the expected key "
         "authorization.",
         ["rfc8555#0068"]),
    33: ("The invalid_grant error means the provided authorization grant (for example, an "
         "authorization code or resource owner credentials) or refresh token is invalid, expired, "
         "or revoked; does not match the redirection URI used in the authorization request; or was "
         "issued to another client.",
         ["rfc6749#0043"]),
    34: ("PKCE (Proof Key for Code Exchange) protects the authorization code flow used by public "
         "clients against the authorization code interception attack. The client creates a secret "
         "code_verifier and sends its transformed value as the code_challenge in the authorization "
         "request; when redeeming the code at the token endpoint it presents the code_verifier, and "
         "the server recomputes and compares it against the earlier code_challenge, returning "
         "invalid_grant if they do not match. An attacker who intercepts only the authorization "
         "code cannot exchange it without the code_verifier.",
         ["rfc7636#0002", "rfc7636#0010"]),
    35: ("A client presents a bearer token to the resource server using one of three methods (and "
         "MUST use only one per request): in the Authorization request header with the \"Bearer\" "
         "scheme (Authorization: Bearer <token>), which is the recommended method; as a "
         "form-encoded body parameter named \"access_token\"; or as a URI query parameter named "
         "\"access_token\".",
         ["rfc6750#0004"]),
    36: ("The Basic authentication scheme transmits credentials as a user-id and password pair "
         "joined by a single colon (\"user-id:password\") and encoded with Base64, sent in the "
         "Authorization header (e.g., \"Authorization: Basic <base64>\"). Because the credentials "
         "are effectively cleartext, Basic is not secure unless used over a secure transport such "
         "as TLS.",
         ["rfc7617#0004"]),
    37: ("The server-specified nonce seeds the computation of the client's Digest response value "
         "and is the scheme's main defense against replay attacks. The server can make the nonce "
         "limited-use — bound to a particular client, resource, time window, or number of uses "
         "(e.g., by encoding a timestamp, client IP, resource ETag, and a private server key) — so "
         "that a captured response cannot be successfully replayed later.",
         ["rfc7616#0028", "rfc7616#0029"]),
    38: ("RFC 7519 defines a set of registered claim names (all short, none mandatory to use): "
         "\"iss\" (issuer), \"sub\" (subject), \"aud\" (audience), \"exp\" (expiration time), "
         "\"nbf\" (not before), \"iat\" (issued at), and \"jti\" (JWT ID).",
         ["rfc7519#0012"]),
    39: ("The JWS Compact Serialization represents a JWS as three base64url-encoded parts joined by "
         "periods: BASE64URL(UTF8(JWS Protected Header)) || '.' || BASE64URL(JWS Payload) || '.' || "
         "BASE64URL(JWS Signature).",
         ["rfc7515#0008"]),
    40: ("In a JWS, the \"alg\" (algorithm) Header Parameter identifies the cryptographic algorithm "
         "used to secure (sign or MAC) the JWS; the JWS Signature is not valid if the \"alg\" value "
         "is not a supported algorithm. Its value is a case-sensitive ASCII string, typically one "
         "registered in the IANA JSON Web Signature and Encryption Algorithms registry.",
         ["rfc7515#0011"]),
    41: ("base64url uses the URL- and filename-safe alphabet from Section 5 of RFC 4648: its 62nd "
         "and 63rd characters are '-' and '_' instead of standard base64's '+' and '/'. As used by "
         "JOSE it also omits the trailing '=' padding and contains no line breaks or whitespace, so "
         "the encoded value is safe to place in URLs and filenames.",
         ["rfc7515#0006", "rfc7515#0081"]),
    42: ("Per RFC 8259, JSON can represent four primitive types — strings, numbers, booleans, and "
         "null — and two structured types: objects (an unordered collection of name/value pairs, "
         "where each name is a string) and arrays (an ordered sequence of values).",
         ["rfc8259#0004"]),
    43: ("RFC 3339 defines an Internet date/time timestamp as date-time = full-date \"T\" "
         "full-time, where full-date is date-fullyear \"-\" date-month \"-\" date-mday (a four-digit "
         "year with two-digit month and day) and full-time is a partial-time (hh:mm:ss with an "
         "optional fractional second) followed by a time-offset that is either \"Z\" (UTC) or "
         "±hh:mm. For example: 1985-04-12T23:20:50.52Z.",
         ["rfc3339#0007"]),
    44: ("An A resource record provides the 32-bit IPv4 address of a host: its RDATA is a single "
         "32-bit Internet address. A host with multiple addresses has multiple A records, and A "
         "records cause no additional section processing.",
         ["rfc1035#0021"]),
    45: ("The Name Error response code (RCODE 3, commonly called NXDOMAIN) indicates that the domain "
         "name referenced in the query does not exist. A resolver that receives such a name-error "
         "indication may cache the negative result for the accompanying TTL and, during that "
         "period, assume the name does not exist without re-querying authoritative servers.",
         ["rfc1034#0028"]),
    46: ("In DNS over HTTPS (DoH) each DNS query/response pair is mapped onto a single HTTP exchange "
         "carried over an https URI (thus protected by TLS). The DNS message uses the "
         "\"application/dns-message\" media type and can be sent either with an HTTP POST (the DNS "
         "message in the request body) or an HTTP GET (the base64url-encoded DNS message supplied "
         "in the \"dns\" query parameter).",
         ["rfc8484#0003", "rfc8484#0007"]),
    47: ("DNS over TLS uses TCP port 853 by default: a client wanting DNS-over-TLS privacy MUST open "
         "a TCP connection to port 853 (unless it has agreed another non-53 port with the server) "
         "and begin with a TLS handshake. Port 853 MUST NOT be used to carry cleartext DNS.",
         ["rfc7858#0006"]),
}


def _clean(answer: str) -> str:
    return _CITE_MARKER.sub("", answer).strip()


def main() -> int:
    seeds = [json.loads(l) for l in SEED_PATH.open(encoding="utf-8")]
    drafts = {d["question"]: d for d in (json.loads(l) for l in DRAFT_PATH.open(encoding="utf-8"))}
    id2text = {
        c.chunk_id: c.text
        for c in (Chunk.model_validate_json(l) for l in CHUNKS_PATH.open(encoding="utf-8"))
    }

    items: list[GroundTruthItem] = []
    problems: list[str] = []
    for i, seed in enumerate(seeds):
        q = seed["question"]
        if i in AUTHORED:
            answer, ctx_ids = AUTHORED[i]
        else:
            d = drafts.get(q)
            if not d or d["_meta"]["status"] != "ok":
                problems.append(f"[{i}] no authored entry and no ok draft: {q[:60]}")
                continue
            answer, ctx_ids = d["reference_answer"], d["_meta"]["cited_chunk_ids"]

        missing = [cid for cid in ctx_ids if cid not in id2text]
        if missing:
            problems.append(f"[{i}] missing chunk_ids {missing}: {q[:50]}")
            continue
        if not ctx_ids:
            problems.append(f"[{i}] no relevant_contexts: {q[:50]}")
            continue

        items.append(GroundTruthItem(
            question=q,
            reference_answer=_clean(answer),
            relevant_contexts=[id2text[cid] for cid in ctx_ids],
        ))

    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  " + p)

    with GROUND_TRUTH_PATH.open("w", encoding="utf-8") as out:
        for it in items:
            out.write(it.model_dump_json() + "\n")
    print(f"\nwrote {len(items)} items -> {GROUND_TRUTH_PATH}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
