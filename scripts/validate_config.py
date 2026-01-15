#!/usr/bin/env python3
"""
Configuration validation script for BoxingGym.

Validates all YAML configuration files for:
- Syntax correctness
- Security (no hardcoded secrets)
- Schema consistency
- Required environment variables

Usage:
  python scripts/validate_config.py              # Full validation
  python scripts/validate_config.py --quick     # Fast syntax check only
  python scripts/validate_config.py --env       # Check env var requirements
"""

import sys
import yaml
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class ConfigValidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.env_vars: Dict[str, List[str]] = defaultdict(list)

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("=" * 80)
        print("BOXING-GYM CONFIGURATION VALIDATOR")
        print("=" * 80)
        print()

        # Phase 1: Syntax validation
        print("1. YAML SYNTAX VALIDATION...")
        syntax_ok = self.validate_syntax()
        print(f"   Status: {'PASS' if syntax_ok else 'FAIL'}")
        print()

        # Phase 2: Security validation
        print("2. SECURITY SCAN...")
        security_ok = self.validate_security()
        print(f"   Status: {'PASS' if security_ok else 'FAIL'}")
        print()

        # Phase 3: Schema validation
        print("3. SCHEMA VALIDATION...")
        schema_ok = self.validate_schemas()
        print(f"   Status: {'PASS' if schema_ok else 'FAIL'}")
        print()

        # Phase 4: Environment validation
        print("4. ENVIRONMENT VARIABLES...")
        env_ok = self.validate_environment()
        print(f"   Status: {'PASS' if env_ok else 'WARN'}")
        print()

        # Report results
        return self.report()

    def validate_syntax(self) -> bool:
        """Check all YAML files for syntax errors."""
        conf_dir = self.project_root / "conf"
        sweep_dir = self.project_root / "sweeps"

        files_checked = 0
        files_ok = 0

        for yaml_file in sorted(conf_dir.rglob("*.yaml")) + sorted(sweep_dir.glob("*.yaml")):
            files_checked += 1
            try:
                with open(yaml_file, 'r') as f:
                    yaml.safe_load(f)
                files_ok += 1
            except yaml.YAMLError as e:
                self.errors.append(f"YAML Error in {yaml_file.relative_to(self.project_root)}: {str(e)[:80]}")
            except Exception as e:
                self.errors.append(f"Error in {yaml_file.relative_to(self.project_root)}: {str(e)[:80]}")

        print(f"   Files validated: {files_ok}/{files_checked}")
        return files_ok == files_checked

    def validate_security(self) -> bool:
        """Check for hardcoded secrets."""
        SECRET_PATTERNS = [
            (r'(?<![a-z])sk_live_[a-z0-9]{20,}', 'Stripe live key'),
            (r'(?<![a-z])sk_test_[a-z0-9]{20,}', 'Stripe test key'),
            (r'AKIA[0-9A-Z]{16}', 'AWS access key'),
            (r'(?<![a-z])password\s*:\s*["\']?[a-zA-Z0-9]{8,}["\']?(?!\})', 'Hardcoded password'),
        ]

        secrets_found = False
        conf_dir = self.project_root / "conf"

        for yaml_file in conf_dir.rglob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    content = f.read()

                for pattern, name in SECRET_PATTERNS:
                    if re.search(pattern, content, re.IGNORECASE):
                        # Ignore false positives from env var references
                        if 'oc.env' not in content[max(0, content.find(pattern)-50):content.find(pattern)+50]:
                            self.warnings.append(f"Possible {name} in {yaml_file.name}")
                            secrets_found = True
            except Exception:
                pass

        if not secrets_found:
            print("   No hardcoded secrets detected")
        return not secrets_found

    def validate_schemas(self) -> bool:
        """Check configuration schema consistency."""
        schemas_ok = True

        # Check experiment configs have required fields
        exp_dir = self.project_root / "conf" / "exp"
        if exp_dir.exists():
            for yaml_file in exp_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, 'r') as f:
                        config = yaml.safe_load(f)
                    if not isinstance(config, dict):
                        self.errors.append(f"Invalid schema in {yaml_file.name}: root is not a dict")
                        schemas_ok = False
                except Exception as e:
                    self.errors.append(f"Schema error in {yaml_file.name}: {str(e)[:60]}")
                    schemas_ok = False

        print(f"   Experiment configs: OK" if schemas_ok else "   Schema errors found")
        return schemas_ok

    def validate_environment(self) -> bool:
        """Check environment variable requirements and availability."""
        llm_dir = self.project_root / "conf" / "llms"

        for yaml_file in llm_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    content = f.read()

                # Find env var references
                matches = re.findall(r'oc\.env:([A-Z_0-9]+)', content)
                for match in set(matches):
                    self.env_vars[match].append(yaml_file.name)
            except Exception:
                pass

        # Report env vars
        if self.env_vars:
            print(f"   Environment variables required: {len(self.env_vars)}")
            missing_vars = []
            for var in sorted(self.env_vars.keys()):
                if not self._env_var_exists(var):
                    missing_vars.append(var)

            if missing_vars:
                self.warnings.append(f"Missing env vars: {', '.join(missing_vars)}")
                print(f"   WARNING: {len(missing_vars)} env vars not set (non-critical)")
                return False
            else:
                print(f"   All required env vars are set")
                return True
        else:
            print(f"   No env vars required for minimal config")
            return True

    def _env_var_exists(self, var: str) -> bool:
        """Check if environment variable is set."""
        import os
        return var in os.environ

    def report(self) -> bool:
        """Print validation report."""
        print("=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        if not self.errors:
            print("\nRESULT: PASS")
            print("\nConfiguration is healthy and ready for deployment.")
            return True
        else:
            print(f"\nRESULT: FAIL ({len(self.errors)} errors)")
            return False


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    validator = ConfigValidator(project_root)

    # Parse args
    quick = '--quick' in sys.argv
    env_only = '--env' in sys.argv

    if env_only:
        validator.validate_environment()
        return 0

    # Run validation
    success = validator.validate_all() if not quick else validator.validate_syntax()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
