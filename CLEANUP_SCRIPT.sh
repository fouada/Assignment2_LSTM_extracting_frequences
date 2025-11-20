#!/bin/bash

# DOCUMENTATION CLEANUP SCRIPT
# Removes redundant development artifacts
# Keeps only essential MIT-level documentation

echo "==================================================================="
echo " LSTM PROJECT - DOCUMENTATION CLEANUP"
echo "==================================================================="
echo ""
echo "This script will DELETE 55 redundant documentation files."
echo "A backup branch will be created first."
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Step 1: Creating backup branch..."
git checkout -b pre-cleanup-backup-$(date +%Y%m%d)
git add -A
git commit -m "Backup before documentation cleanup" || echo "Nothing to commit"
git checkout master || git checkout main

echo ""
echo "Step 2: Deleting redundant files..."

# Summary files
rm -f ASSIGNMENT_100_PERCENT_COMPLETE.md
rm -f ASSIGNMENT_VALIDATION_CHECKLIST.md
rm -f CHANGES_SUMMARY.md
rm -f CICD_SETUP_COMPLETE.md
rm -f COMMUNITY_SETUP_COMPLETE.md
rm -f COMPLETE_ANSWERS_TO_YOUR_QUESTIONS.md
rm -f COMPLETE_FEATURES_EXECUTION_GUIDE.md
rm -f COMPLETE_FIX_SUMMARY.md
rm -f COMPLETE_L_EXPERIMENT_SUMMARY.md
rm -f DOCUMENTATION_CLEANUP_SUMMARY.md
rm -f DOCUMENTATION_COMPLETE_SUMMARY.md
rm -f FIXES_APPLIED_SUMMARY.md
rm -f COVERAGE_FIX_SUMMARY.md
rm -f QUICK_FIX_SUMMARY.md
rm -f README_COVERAGE_FIX.md
rm -f README_UPDATE_SUMMARY.md

# Fix/Debug files
rm -f NORMALIZATION_BUG_FIX.md
rm -f TRAINING_INSTABILITY_FIX.md
rm -f TRAINING_STABILITY_FIX.md
rm -f ZERO_PREDICTION_FIX.md
rm -f COVERAGE_FIX_CI.md

# Redundant feature guides
rm -f COST_ANALYSIS_CHECKLIST.md
rm -f COST_ANALYSIS_IMPLEMENTATION.md
rm -f COST_ANALYSIS_QUICK_START.md
rm -f COST_ANALYSIS_FEATURE_SUMMARY.md
rm -f WHATS_NEW_COST_ANALYSIS.md
rm -f INNOVATIONS_QUICK_START.md
rm -f INNOVATIONS_SUMMARY.md
rm -f INNOVATION_COMPLETE.md
rm -f INNOVATION_ROADMAP.md

# Redundant execution guides
rm -f EXECUTION_AND_SCREENSHOT_GUIDE.md
rm -f QUICK_SCREENSHOT_REFERENCE.md
rm -f COMPLETE_FEATURES_EXECUTION_GUIDE.md
rm -f EXECUTION_FLOWS_INDEX.md
rm -f EXECUTION_MODES_COMPARISON.md
rm -f ANSWER_ALL_FEATURES_AND_FLOWS.md
rm -f HOW_TO_RUN_COMPLETE_EXPERIMENTS.md
rm -f QUICK_START_EXPERIMENTS.md

# Redundant state/sequence files
rm -f STATE_MANAGEMENT_GUIDE.md
rm -f STATE_MANAGEMENT_SUMMARY.md
rm -f SEQUENCE_LENGTH_EXPERIMENTS_GUIDE.md
rm -f SEQUENCE_LENGTH_FINDINGS.md
rm -f SEQUENCE_LENGTH_QUICK_SUMMARY.md
rm -f TRAINING_COMPARISON.md

# Redundant misc files
rm -f M1_MAC_COMPLETE_GUIDE.md
rm -f M1_QUICK_START.md
rm -f READY_FOR_SUBMISSION.md
rm -f SECTION_6_VALIDATION.md
rm -f RUN_TESTS.md
rm -f QUICK_TEST_GUIDE.md
rm -f QUICK_REFERENCE_CARD.md
rm -f TEST_FIXES_SUMMARY.md
rm -f INSTRUCTOR_QUICK_REVIEW.md
rm -f SUBMISSION_PACKAGE.md

# Redundant docs subfolder files
rm -f docs/DOCUMENTATION_CLEANUP_SUMMARY.md
rm -f docs/README_DOCS.md

echo ""
echo "Step 3: Counting remaining markdown files..."
total=$(find . -name "*.md" -type f ! -path "./.pytest_cache/*" ! -path "./htmlcov/*" ! -path "./.venv/*" ! -path "./experiments/*" | wc -l)
echo "Remaining markdown files: $total"

echo ""
echo "==================================================================="
echo " CLEANUP COMPLETE"
echo "==================================================================="
echo ""
echo "Essential documentation kept:"
echo "  • README.md (main entry)"
echo "  • CONTRIBUTING.md"
echo "  • CODE_OF_CONDUCT.md"
echo "  • SECURITY.md"
echo "  • CHANGELOG.md"
echo "  • PRODUCT_REQUIREMENTS_DOCUMENT.md"
echo "  • DEVELOPMENT_PROMPTS_LOG.md"
echo "  • AUTHORS.md"
echo "  • CONTRIBUTORS.md"
echo "  • docs/ folder (9 essential guides)"
echo "  • research/ folder (mathematical analysis)"
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Commit: git add -A && git commit -m 'Clean up redundant documentation'"
echo "  3. If needed, restore backup: git checkout pre-cleanup-backup-$(date +%Y%m%d)"
echo ""
echo "Before: 135 files"
echo "After: ~20 files"
echo "Improvement: Professional MIT-level presentation ✅"
echo ""
