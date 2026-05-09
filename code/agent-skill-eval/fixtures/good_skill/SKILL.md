---
name: receipt-ingest
description: Add receipts to a household expense tracker from images, PDFs, or plain text.
---

# Receipt Ingest Skill

## When to use

Use this skill when the user sends a receipt, invoice, utility bill, or asks to record an expense.

## When not to use

Do not use it for unrelated banking questions or investment advice.

## Quick start

1. Read the input image/PDF/text.
2. Extract merchant, date, total, currency, category, and submitted_by.
3. Check for duplicates before writing.
4. Append one normalized row to the tracker.
5. Verify the row was written and report the result.

## Inputs

- receipt image, PDF, or plain text
- optional submitted_by value
- optional category override

## Output

Return the recorded merchant, amount, date, category, and duplicate status.

## Examples

## Common commands

```bash
python add_receipt.py --input receipt.jpg --submitted-by grzegorz --dry-run
python add_receipt.py --input receipt.jpg --submitted-by grzegorz
```

## Safety

Ask for confirmation before deleting, overwriting, or exporting private financial data. Never print tokens,
passwords, or credentials. If a receipt contains sensitive private information, keep it local.

## Failure recovery

If OCR fails, retry once with a higher-resolution conversion. If the tracker is locked or unavailable,
write a local pending JSON file and tell the user what still needs to be synced.

## External-system discipline

Batch writes when possible, serialize updates to avoid conflicts, and respect 429 Retry-After responses.
