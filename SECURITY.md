# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **zaudrehman@gmail.com**

Include the following information:

- Type of issue (e.g., false positive amplification, hash collision, unsound unsafe)
- Full paths of source file(s) related to the manifestation of the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Status Updates**: Every week until resolved
- **Fix Timeline**: Varies based on severity

## Security Best Practices

When using BloomCraft:

- Always validate input before inserting into filters
- Use appropriate false positive rates for your security requirements
- Consider the implications of FPR in security-critical applications
- Keep BloomCraft and all dependencies up to date
- Use secure hash functions (default SipHash-1-3 is recommended)

## Known Security Considerations

### False Positives

Bloom filters can have false positives. Do NOT use Bloom filters alone for:

- Authentication decisions
- Security token validation
- Cryptographic key verification
- Payment processing

Always use Bloom filters as a preliminary filter with secondary verification.

Thank you for helping keep BloomCraft secure!
