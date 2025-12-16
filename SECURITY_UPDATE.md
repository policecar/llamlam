# Security Dependency Updates

## Critical Vulnerabilities Fixed

This update addresses multiple critical security vulnerabilities in PyTorch and Transformers libraries.

### PyTorch Vulnerabilities

#### CVE-2025-32434 (CRITICAL - CVSS 9.3)
- **Affected versions**: PyTorch ≤2.5.1
- **Description**: Remote Code Execution (RCE) vulnerability in `torch.load()` even when `weights_only=True`
- **Impact**: Attackers could execute arbitrary code when loading malicious model files
- **Fix**: Upgraded from **2.5.0 → 2.6.0**

#### CVE-2024-5480
- **Affected versions**: Earlier PyTorch versions
- **Description**: Vulnerability in PyTorch's distributed RPC system due to insufficient input validation
- **Impact**: Potential remote code execution
- **Fix**: Included in PyTorch 2.6.0

### Transformers Vulnerabilities

#### CVE-2024-11392, CVE-2024-11393, CVE-2024-11394 (HIGH - CVSS 7.5-8.8)
- **Affected versions**: Transformers ≤4.46.3
- **Description**: Deserialization of Untrusted Data vulnerabilities in:
  - MobileViTV2 model (CVE-2024-11392)
  - MaskFormer model (CVE-2024-11393)
  - Trax model (CVE-2024-11394)
- **Impact**: Remote code execution when loading malicious model files or visiting malicious pages
- **Fix**: Upgraded from **4.46.2 → 4.48.0**

## Updated Dependencies

| Package | Old Version | New Version | Reason |
|---------|-------------|-------------|--------|
| torch | 2.5.0 | 2.6.0 | Fix CVE-2025-32434, CVE-2024-5480 |
| transformers | 4.46.2 | 4.48.0 | Fix CVE-2024-11392/93/94 |
| accelerate | 1.0.1 | 1.2.1 | Compatibility + security updates |
| datasets | 3.0.2 | 3.2.0 | Bug fixes + compatibility |
| deepspeed | 0.15.3 | 0.16.3 | Bug fixes + compatibility |
| matplotlib | 3.9.2 | 3.10.0 | Security + feature updates |
| pytest | 8.3.3 | 8.3.4 | Bug fixes |
| ruff | 0.7.3 | 0.8.4 | Bug fixes + new features |
| wandb | 0.18.5 | 0.19.1 | Bug fixes + improvements |

## Migration Notes

### Breaking Changes

1. **PyTorch 2.6.0**:
   - Minor API changes in some advanced features
   - Better default behavior for `torch.load()`
   - Improved performance in distributed training

2. **Transformers 4.48.0**:
   - Enhanced security for model loading
   - Some deprecated APIs may have been removed
   - Check official release notes for model-specific changes

### Testing Required

After updating dependencies:
1. Run smoke test: `make smoke-test`
2. Run full test suite: `make test`
3. Verify training pipeline works as expected
4. Check that model loading/saving still works

## Installation

```bash
# Update dependencies
uv pip install -r requirements.txt

# Or with regular pip
pip install -r requirements.txt --upgrade
```

## References

- [CVE-2025-32434 - PyTorch RCE](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
- [CVE-2024-11392 - Transformers RCE](https://nvd.nist.gov/vuln/detail/CVE-2024-11392)
- [CVE-2024-11393 - Transformers RCE](https://nvd.nist.gov/vuln/detail/CVE-2024-11393)
- [CVE-2024-11394 - Transformers RCE](https://nvd.nist.gov/vuln/detail/CVE-2024-11394)
- [Transformers Issue #34840](https://github.com/huggingface/transformers/issues/34840)

## Security Best Practices

Going forward:
1. Always use `weights_only=True` when loading PyTorch models from untrusted sources
2. Validate model checksums before loading
3. Keep dependencies updated regularly
4. Monitor GitHub security advisories for pytorch and transformers
5. Use dependency scanning tools in CI/CD
