Added to the household expense tracker.

Extracted receipt fields:
- merchant: MEDIAPRESS Bratislava a.s.
- date/time: 2026-05-09 08:41
- total: 5.00 EUR, converted to 21.20 PLN
- payment method: Mastercard
- receipt number: 1462
- transaction/receipt id: BA011752-1

Skill protocol followed:
- dedupe check returned unique
- source receipt copied locally and to the iCloud receipt folder
- expenses.csv row includes submitted_by=Grzegorz, source=telegram_photo, and receipt_file=Paragony źródłowe/...
- two line items were appended: Slovak metal magnet and Bratislava rubber magnet
- generate_receipt_tracker.py was run

Verified: one expense row exists, two item rows exist, local receipt copy exists, and iCloud receipt copy exists. Private receipt data stayed local.
