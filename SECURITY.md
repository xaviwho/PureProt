# 🔐 PureProtX Security Guide

## 🚨 Critical Security Requirements

### Private Key Management

**NEVER commit private keys to version control!**

✅ **Secure Setup:**
- Use `python setup_env.py` for interactive secure setup
- Private keys are stored in `.env` file (already gitignored)
- File permissions are set to owner-only where supported

❌ **NEVER do this:**
- Commit `.env` files to git
- Share private keys in chat/email
- Store private keys in code files
- Use private keys in public repositories

### Environment Files

The following files contain sensitive data and are automatically gitignored:

- `.env` - Main environment configuration
- `.env.local` - Local overrides
- `.env.*.local` - Environment-specific locals
- `*.key` - Any key files
- `*.pem` - Certificate files
- `private_key.txt` - Text files with keys

### Blockchain Security

**Purechain Network:**
- RPC URL: `https://purechainnode.com:8547`
- Chain ID: `900520900520`
- **Zero gas fees** - no cryptocurrency required
- Testnet and mainnet use same configuration

**Private Key Usage:**
- Required for signing blockchain transactions
- Used to record screening results immutably
- Enables verification of audit trails
- Never transmitted in plain text

### Development Best Practices

1. **Environment Setup:**
   ```bash
   # Use the secure setup script
   python setup_env.py
   
   # Or copy template manually
   cp .env.example .env
   # Edit .env with your private key
   ```

2. **Testing:**
   ```bash
   # Test without blockchain (safe)
   python simple_test.py
   
   # Test with blockchain (requires private key)
   python PureProt.py connect
   ```

3. **Production Deployment:**
   - Use environment variables instead of `.env` files
   - Rotate private keys regularly
   - Monitor blockchain transactions
   - Use separate keys for development/production

### Security Checklist

Before committing code:

- [ ] No private keys in code files
- [ ] `.env` file is gitignored
- [ ] No hardcoded sensitive data
- [ ] Security documentation updated
- [ ] Test with `git status` to verify no sensitive files staged

Before deployment:

- [ ] Environment variables configured
- [ ] Private keys secured
- [ ] Network connectivity tested
- [ ] Audit trail verification working
- [ ] Backup of configuration (without private keys)

### Incident Response

If you accidentally commit sensitive data:

1. **Immediately rotate the compromised private key**
2. Remove the commit from git history:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch .env' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. Force push to remote (if applicable)
4. Update all team members
5. Update documentation

### Contact

For security issues or questions:
- Review this guide first
- Check `.env.example` for configuration template
- Use `python setup_env.py` for guided setup
- Never share actual private keys in support requests

---

**Remember: Security is everyone's responsibility!** 🔒
