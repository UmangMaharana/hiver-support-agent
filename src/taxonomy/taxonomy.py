INTENTS = {
    "delivery_issue": "Problems with an existing delivery being late, missing, damaged, or otherwise not delivered as expected.",
    "delivery_slots": "Questions or problems about booking, changing, or availability of delivery slots.",
    "online_website_issue": "Technical problems with Tesco website, app, online shopping systems, or related digital functionality.",
    "online_order_issue": "Problems with an existing online order, including missing/wrong items, cancellation, or order processing.",
    "click_collect": "Questions or problems specifically involving Click & Collect orders or collection.",
    "product_availability": "Questions about whether, where, or when a product is available or back in stock.",
    "product_quality_safety": "Spoiled, expired, contaminated, damaged, defective, or otherwise unsafe/poor-quality products.",
    "pricing_offers": "Incorrect prices, discounts, promotions, price labels, or offer-related issues.",
    "refund_return_compensation": "Requests or questions about refunds, returns, replacements, reimbursement, or compensation.",
    "store_service_complaint": "Complaints about store staff, store experience, facilities, checkout, or in-store service.",
    "product_information": "Questions about ingredients, allergens, dietary suitability, pesticides, or other product information.",
    "clubcard_account": "Clubcard, account access, vouchers, or account-related issues.",
    "other": "Social chatter, praise, irrelevant content, vague requests, or issues outside the supported taxonomy.",
}

INTENT_NAMES = list(INTENTS.keys())