# Tesco Support Agent — Failure Analysis

This report is generated directly from the frozen evaluation artifacts. It is intended to support manual review and selection of representative cases for the final project report.

## 1. Executive Summary

- Intent classification: **101/250 errors (40.4%)**.
- The dominant classification problem is confusion among `other`, `store_service_complaint`, and `product_quality_safety`.
- Gemini escalation produced **7 false positives and 1 false negative** on the 50-example human benchmark.
- The deterministic rule baseline produced **1 false positive and 23 false negatives**.
- The reply judge flagged **4/250 responses (1.6%)** for unsupported claims.
- **15 cases** had an incorrect predicted intent but still received a high-quality reply (judge score >= 4/5).
- **2 cases** had the correct intent but a low-quality reply (judge score <= 3/5).

## 2. Intent Classification Failures

The most frequent confusion pairs are:

- `other` → `product_quality_safety`: other -> product_quality_safety (18 cases)
- `other` → `store_service_complaint`: other -> store_service_complaint (8 cases)
- `store_service_complaint` → `product_quality_safety`: store_service_complaint -> product_quality_safety (8 cases)
- `other` → `pricing_offers`: other -> pricing_offers (4 cases)
- `store_service_complaint` → `pricing_offers`: store_service_complaint -> pricing_offers (4 cases)
- `other` → `online_website_issue`: other -> online_website_issue (3 cases)
- `product_information` → `product_quality_safety`: product_information -> product_quality_safety (3 cases)
- `store_service_complaint` → `product_availability`: store_service_complaint -> product_availability (3 cases)
- `pricing_offers` → `product_availability`: pricing_offers -> product_availability (2 cases)
- `other` → `product_availability`: other -> product_availability (2 cases)
- `online_website_issue` → `pricing_offers`: online_website_issue -> pricing_offers (2 cases)
- `product_availability` → `store_service_complaint`: product_availability -> store_service_complaint (2 cases)
- `store_service_complaint` → `delivery_issue`: store_service_complaint -> delivery_issue (2 cases)
- `product_quality_safety` → `other`: product_quality_safety -> other (2 cases)
- `product_quality_safety` → `pricing_offers`: product_quality_safety -> pricing_offers (2 cases)

### Representative intent errors

### GOLD_028
**Gold intent:** `product_availability`  **Predicted:** `online_order_issue`

**Customer:**
> hey @Tesco why no courgettes for online customers been trying for three weeks. Am now looking at @sainsburys and they have them may switch

**Generated reply:**
> Hi there! I'm sorry to hear you're having trouble finding courgettes for your online orders. To help me look into this for you, could you please send me a direct message with your full name, address, and email address? Thank you!

**Failure:** product_availability -> online_order_issue
**Judge reason:** The response fails to address the customer's general question about courgette availability for online customers, instead immediately asking for personal details for an investigation the bot cannot perform.

### GOLD_029
**Gold intent:** `other`  **Predicted:** `pricing_offers`

**Customer:**
> I just got told off for putting the Christmas radio station on at work #bahhumbug

**Generated reply:**
> I'm sorry to hear you were told off at work. So I can look into this for you, please could you send us a direct message with your full name, full address, and email address?

**Failure:** other -> pricing_offers
**Judge reason:** The model completely misunderstands the customer's message, assuming a Tesco staff issue when the customer is venting about their own workplace. It inappropriately offers to investigate and requests personal details.

### GOLD_011
**Gold intent:** `online_order_issue`  **Predicted:** `clubcard_account`

**Customer:**
> Hi @Tesco I've a query about Christmas dinner orders, who should I ask?

**Generated reply:**
> Hi there! To help you with your query about Christmas dinner orders, could you please tell me a bit more about what you'd like to know?

**Failure:** online_order_issue -> clubcard_account
**Judge reason:** The response doesn't directly answer 'who' to ask but appropriately seeks more information to understand the customer's specific query.

### GOLD_002
**Gold intent:** `other`  **Predicted:** `product_quality_safety`

**Customer:**
> Seriously @Tesco  since when have you started cutting crusts of children’s sandwiches. Can’t believe your encouraging this!!! #parentinghell https://t.co/bNhABXSdK6

**Generated reply:**
> Hi there, thanks for reaching out. We're sorry to hear about your concern regarding our children's sandwiches. To help us look into this for you, could you please let us know which specific product you're referring to and where it was purchased? We'll then pass this on to our team.

**Failure:** other -> product_quality_safety
**Judge reason:** The response misinterprets the customer's general policy question as a specific product complaint, asking for irrelevant details. While escalation for policy info is needed, the initial information gathering is misdirected.

### GOLD_038
**Gold intent:** `product_information`  **Predicted:** `product_quality_safety`

**Customer:**
> @Tesco the cal content per sachet of your "easy oats, blueberry &amp; apple" is that before milk is added? or per made up sachet? its not clear!

**Generated reply:**
> Thanks for reaching out about the calorie content on our Easy Oats, Blueberry & Apple sachets. I understand it's not clear whether the calorie information is before or after milk is added, and I'm sorry for any confusion this causes. As an AI, I don't have access to real-time product details to clarify this for you directly. I can pass on your feedback about the labeling to our team for review. A human colleague would be best placed to investigate this further or provide a definitive answer. Would you like me to escalate this so they can look into it?

**Failure:** product_information -> product_quality_safety
**Judge reason:** The AI correctly identifies the customer's query but claims it cannot access product details, despite historical examples showing this information was directly provided by agents. This leads to unnecessary escalation for a static product detail.

### GOLD_006
**Gold intent:** `other`  **Predicted:** `delivery_slots`

**Customer:**
> I do love  Christmas films!

**Generated reply:**
> It's lovely to hear you enjoy Christmas films! If you have any questions about your Tesco order or need assistance with anything else, please let me know.

**Failure:** other -> delivery_slots
**Judge reason:** The response acknowledges the customer's statement but then immediately pivots to a generic support offer, missing the opportunity for conversational engagement shown in historical examples.

### GOLD_219
**Gold intent:** `online_order_issue`  **Predicted:** `product_quality_safety`

**Customer:**
> @Tesco oi oi Tesco, there’s a cheap brand intruder in my shopping! That’s not what I paid for... https://t.co/wnIHWLWdqS

**Generated reply:**
> I'm sorry to hear you received an incorrect item in your shopping. A colleague will be able to look into this for you and help resolve the issue.

**Failure:** online_order_issue -> product_quality_safety
**Judge reason:** Correctly identifies the issue and escalates, but could be more helpful by asking for necessary order details upfront to expedite resolution.

### GOLD_212
**Gold intent:** `other`  **Predicted:** `product_information`

**Customer:**
> @Tesco is this a scam?? https://t.co/CQq0cK0Zui

**Generated reply:**
> Thanks for reaching out about this link. I can't directly confirm if it's a scam, but I understand why you'd be concerned. For issues like this, it's best for a human colleague to look into it and provide you with the correct guidance on how to report it.

**Failure:** other -> product_information
**Judge reason:** The response correctly identifies its limitation and escalates, but misses the opportunity to provide the reporting email address seen in historical examples.

### GOLD_010
**Gold intent:** `product_availability`  **Predicted:** `online_website_issue`

**Customer:**
> @Tesco are you able to confirm for me what time the new Pokemon games go on sale tomorrow?

**Generated reply:**
> I'm sorry, I don't have information on the exact release time for new Pokemon games.

**Failure:** product_availability -> online_website_issue
**Judge reason:** The response directly addresses the customer's query by stating its limitation, which is factual and safe. Escalation is appropriate as this specific information would likely require human intervention.

### GOLD_218
**Gold intent:** `pricing_offers`  **Predicted:** `delivery_issue`

**Customer:**
> @Tesco signed up for delivery saver free trial and was still charged for delivery at checkout on first order??

**Generated reply:**
> I'm sorry to hear you were charged for delivery on your first order after signing up for the Delivery Saver free trial. I don't have access to customer accounts or order details to investigate this directly. A colleague will be able to look into your Delivery Saver subscription and the charge on your order. It would be helpful if you could provide your full name, address, email, and order number when you speak with them.

**Failure:** pricing_offers -> delivery_issue
**Judge reason:** The response correctly identifies its limitations and appropriately escalates, providing useful information for the customer. It could be improved by offering general explanations from historical examples before immediate escalation.

## 3. Retrieval Failures

The following are among the lowest-scoring human-evaluated retrieval pairs.

### GOLD_003
**Human retrieval score:** 0.0

**Customer:**
> @Tesco estimate i've spent £78k with you but told today to pick my own shopping up from store means that figure will not be increasing!

### GOLD_007
**Human retrieval score:** 0.0

**Customer:**
> @Tesco Hi, is there a limit to how much wine you can buy in one order for Home Delivery?

### GOLD_012
**Human retrieval score:** 0.0

**Customer:**
> But in @Tesco Apple Pay is still limited to £30. Such a shame. No limit at @Morrisons I’m pleased to say :-) https://t.co/7sb1KkGWvS

### GOLD_039
**Human retrieval score:** 0.0

**Customer:**
> @Tesco Hi, per 100g the asda chicken thigh fillets has 27g protein, sainsburys has 26.2g, but tesco ones only have 18.3g, i normally do my shop in tesco but have to buy chicken thighs elsewhere, my email is __email__, i can email my details, rather not post online

### GOLD_027
**Human retrieval score:** 0.0

**Customer:**
> @Tesco am I right in saying delivery is only happening on 24th Dec around Fulfillment Centres? Mainly London? I've booked for 23rd hoping 24th isn't happening in nottm.

### GOLD_031
**Human retrieval score:** 0.0

**Customer:**
> @Tesco if it had been clear taking the delivery free trial excluded me from the Christmas priority booking I wouldn't have taken you up on it.

### GOLD_023
**Human retrieval score:** 0.0

**Customer:**
> @Tesco How can I reprint my pizza express voucher bought with clubcard as lost original?

### GOLD_023
**Human retrieval score:** 0.0

**Customer:**
> @Tesco How can I reprint my pizza express voucher bought with clubcard as lost original?

### GOLD_111
**Human retrieval score:** 0.0

**Customer:**
> @Tesco hello can you see if Tesco havant has any PlayStation vr in stock and current price cheers

### GOLD_103
**Human retrieval score:** 0.0

**Customer:**
> hi, @Tesco - your smoked bacon lardons are more water than lardon

please stop

## 4. Lowest-Quality Generated Replies

### GOLD_028
**Gold intent:** `product_availability`  **Predicted:** `online_order_issue`

**Customer:**
> hey @Tesco why no courgettes for online customers been trying for three weeks. Am now looking at @sainsburys and they have them may switch

**Generated reply:**
> Hi there! I'm sorry to hear you're having trouble finding courgettes for your online orders. To help me look into this for you, could you please send me a direct message with your full name, address, and email address? Thank you!

**Failure:** Overall quality = 1
**Judge reason:** The response fails to address the customer's general question about courgette availability for online customers, instead immediately asking for personal details for an investigation the bot cannot perform.

### GOLD_029
**Gold intent:** `other`  **Predicted:** `pricing_offers`

**Customer:**
> I just got told off for putting the Christmas radio station on at work #bahhumbug

**Generated reply:**
> I'm sorry to hear you were told off at work. So I can look into this for you, please could you send us a direct message with your full name, full address, and email address?

**Failure:** Overall quality = 1
**Judge reason:** The model completely misunderstands the customer's message, assuming a Tesco staff issue when the customer is venting about their own workplace. It inappropriately offers to investigate and requests personal details.

### GOLD_060
**Gold intent:** `clubcard_account`  **Predicted:** `clubcard_account`

**Customer:**
> @Tesco if I Exchange my club vouchers for pizza of cineworld how long do the exchanges vouchers last for ??

**Generated reply:**
> Hi there! Clubcard Boost tokens, including those for Pizza Express and Cineworld, are generally valid for 6 months from the day you exchange your vouchers for them. I hope this helps!

**Failure:** Overall quality = 2
**Judge reason:** The response directly answers the customer's question but makes a specific factual claim about the validity period that is not supported by the provided historical examples.

### GOLD_002
**Gold intent:** `other`  **Predicted:** `product_quality_safety`

**Customer:**
> Seriously @Tesco  since when have you started cutting crusts of children’s sandwiches. Can’t believe your encouraging this!!! #parentinghell https://t.co/bNhABXSdK6

**Generated reply:**
> Hi there, thanks for reaching out. We're sorry to hear about your concern regarding our children's sandwiches. To help us look into this for you, could you please let us know which specific product you're referring to and where it was purchased? We'll then pass this on to our team.

**Failure:** Overall quality = 3
**Judge reason:** The response misinterprets the customer's general policy question as a specific product complaint, asking for irrelevant details. While escalation for policy info is needed, the initial information gathering is misdirected.

### GOLD_012
**Gold intent:** `pricing_offers`  **Predicted:** `pricing_offers`

**Customer:**
> But in @Tesco Apple Pay is still limited to £30. Such a shame. No limit at @Morrisons I’m pleased to say :-) https://t.co/7sb1KkGWvS

**Generated reply:**
> Thanks for sharing your feedback regarding the Apple Pay limit at Tesco. We appreciate you letting us know about your experience.

**Failure:** Overall quality = 3
**Judge reason:** The response politely acknowledges the customer's feedback but misses the opportunity to provide helpful information about the Apple Pay limit, which has been historically provided by Tesco.

### GOLD_011
**Gold intent:** `online_order_issue`  **Predicted:** `clubcard_account`

**Customer:**
> Hi @Tesco I've a query about Christmas dinner orders, who should I ask?

**Generated reply:**
> Hi there! To help you with your query about Christmas dinner orders, could you please tell me a bit more about what you'd like to know?

**Failure:** Overall quality = 3
**Judge reason:** The response doesn't directly answer 'who' to ask but appropriately seeks more information to understand the customer's specific query.

### GOLD_038
**Gold intent:** `product_information`  **Predicted:** `product_quality_safety`

**Customer:**
> @Tesco the cal content per sachet of your "easy oats, blueberry &amp; apple" is that before milk is added? or per made up sachet? its not clear!

**Generated reply:**
> Thanks for reaching out about the calorie content on our Easy Oats, Blueberry & Apple sachets. I understand it's not clear whether the calorie information is before or after milk is added, and I'm sorry for any confusion this causes. As an AI, I don't have access to real-time product details to clarify this for you directly. I can pass on your feedback about the labeling to our team for review. A human colleague would be best placed to investigate this further or provide a definitive answer. Would you like me to escalate this so they can look into it?

**Failure:** Overall quality = 3
**Judge reason:** The AI correctly identifies the customer's query but claims it cannot access product details, despite historical examples showing this information was directly provided by agents. This leads to unnecessary escalation for a static product detail.

### GOLD_006
**Gold intent:** `other`  **Predicted:** `delivery_slots`

**Customer:**
> I do love  Christmas films!

**Generated reply:**
> It's lovely to hear you enjoy Christmas films! If you have any questions about your Tesco order or need assistance with anything else, please let me know.

**Failure:** Overall quality = 3
**Judge reason:** The response acknowledges the customer's statement but then immediately pivots to a generic support offer, missing the opportunity for conversational engagement shown in historical examples.

### GOLD_219
**Gold intent:** `online_order_issue`  **Predicted:** `product_quality_safety`

**Customer:**
> @Tesco oi oi Tesco, there’s a cheap brand intruder in my shopping! That’s not what I paid for... https://t.co/wnIHWLWdqS

**Generated reply:**
> I'm sorry to hear you received an incorrect item in your shopping. A colleague will be able to look into this for you and help resolve the issue.

**Failure:** Overall quality = 3
**Judge reason:** Correctly identifies the issue and escalates, but could be more helpful by asking for necessary order details upfront to expedite resolution.

### GOLD_212
**Gold intent:** `other`  **Predicted:** `product_information`

**Customer:**
> @Tesco is this a scam?? https://t.co/CQq0cK0Zui

**Generated reply:**
> Thanks for reaching out about this link. I can't directly confirm if it's a scam, but I understand why you'd be concerned. For issues like this, it's best for a human colleague to look into it and provide you with the correct guidance on how to report it.

**Failure:** Overall quality = 3
**Judge reason:** The response correctly identifies its limitation and escalates, but misses the opportunity to provide the reporting email address seen in historical examples.

## 5. Unsupported Claims

Found **4** judge-flagged unsupported-claim cases.

### GOLD_028
**Gold intent:** `product_availability`  **Predicted:** `online_order_issue`

**Customer:**
> hey @Tesco why no courgettes for online customers been trying for three weeks. Am now looking at @sainsburys and they have them may switch

**Generated reply:**
> Hi there! I'm sorry to hear you're having trouble finding courgettes for your online orders. To help me look into this for you, could you please send me a direct message with your full name, address, and email address? Thank you!

**Failure:** Judge flagged unsupported claim
**Judge reason:** The response fails to address the customer's general question about courgette availability for online customers, instead immediately asking for personal details for an investigation the bot cannot perform.

### GOLD_029
**Gold intent:** `other`  **Predicted:** `pricing_offers`

**Customer:**
> I just got told off for putting the Christmas radio station on at work #bahhumbug

**Generated reply:**
> I'm sorry to hear you were told off at work. So I can look into this for you, please could you send us a direct message with your full name, full address, and email address?

**Failure:** Judge flagged unsupported claim
**Judge reason:** The model completely misunderstands the customer's message, assuming a Tesco staff issue when the customer is venting about their own workplace. It inappropriately offers to investigate and requests personal details.

### GOLD_038
**Gold intent:** `product_information`  **Predicted:** `product_quality_safety`

**Customer:**
> @Tesco the cal content per sachet of your "easy oats, blueberry &amp; apple" is that before milk is added? or per made up sachet? its not clear!

**Generated reply:**
> Thanks for reaching out about the calorie content on our Easy Oats, Blueberry & Apple sachets. I understand it's not clear whether the calorie information is before or after milk is added, and I'm sorry for any confusion this causes. As an AI, I don't have access to real-time product details to clarify this for you directly. I can pass on your feedback about the labeling to our team for review. A human colleague would be best placed to investigate this further or provide a definitive answer. Would you like me to escalate this so they can look into it?

**Failure:** Judge flagged unsupported claim
**Judge reason:** The AI correctly identifies the customer's query but claims it cannot access product details, despite historical examples showing this information was directly provided by agents. This leads to unnecessary escalation for a static product detail.

### GOLD_060
**Gold intent:** `clubcard_account`  **Predicted:** `clubcard_account`

**Customer:**
> @Tesco if I Exchange my club vouchers for pizza of cineworld how long do the exchanges vouchers last for ??

**Generated reply:**
> Hi there! Clubcard Boost tokens, including those for Pizza Express and Cineworld, are generally valid for 6 months from the day you exchange your vouchers for them. I hope this helps!

**Failure:** Judge flagged unsupported claim
**Judge reason:** The response directly answers the customer's question but makes a specific factual claim about the validity period that is not supported by the provided historical examples.

## 6. Gemini Escalation False Positives

Found **7** cases where Gemini escalated but the human benchmark said no escalation was needed.

### GOLD_029
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I'm sorry to hear you were told off at work. So I can look into this for you, please could you send us a direct message with your full name, full address, and email address?

**Failure:** Gemini escalated but human said NO

### GOLD_095
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> It's great to hear you're enjoying our new craft beer range! I don't have access to information about future product lines or what will be stocked for Christmas. For specific product requests like a Russian imperial stout, a human colleague would be best placed to pass your feedback on to our buying team.

**Failure:** Gemini escalated but human said NO

### GOLD_036
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> Hi there! I understand you're curious about our new Christmas menu deal and how it compares to last year's. I don't have the specific details about the current menu or its availability right now. A member of our team would be best placed to help you with the most up-to-date information.

**Failure:** Gemini escalated but human said NO

### GOLD_085
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> Hi there, we no longer offer 5p bags as we've introduced new bags for life. These bags are reusable, made from 94% recycled material, and can be replaced for free if they get damaged. A human colleague will be able to provide further context on your concerns.

**Failure:** Gemini escalated but human said NO

### GOLD_038
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> Thanks for reaching out about the calorie content on our Easy Oats, Blueberry & Apple sachets. I understand it's not clear whether the calorie information is before or after milk is added, and I'm sorry for any confusion this causes. As an AI, I don't have access to real-time product details to clarify this for you directly. I can pass on your feedback about the labeling to our team for review. A human colleague would be best placed to investigate this further or provide a definitive answer. Would you like me to escalate this so they can look into it?

**Failure:** Gemini escalated but human said NO

### GOLD_037
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I'm sorry to hear that your salmon wasn't scaled or pin-boned, especially after looking forward to it. I understand your concern about being charged for the weight of bones and scales. To help us look into this for you, could you please provide more details about the product, such as the specific salmon product name and where and when you purchased it? A human colleague will be best placed to investigate this and address your query about pricing and product quality.

**Failure:** Gemini escalated but human said NO

### GOLD_147
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I understand your concern about allergen labeling. I don't have access to specific product ingredient lists or legal requirements for labeling. A human colleague would be best placed to investigate this for you. Could you please let me know which specific product you are referring to?

**Failure:** Gemini escalated but human said NO

## 7. Gemini Escalation False Negatives

Found **1** case(s) where Gemini did not escalate but the human benchmark said escalation was needed.

### GOLD_003
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I'm sorry to hear about your recent experience. So I can look into this for you, please could you DM your full name, full address, email address, which store you visited, and when this happened? Thank you.

**Failure:** Gemini did not escalate but human said YES

## 8. Rule Baseline Failures

- Rule false negatives: **23** in the 50-example human benchmark.
- Rule false positives: **1** in the 50-example human benchmark.

The rule baseline's main weakness is under-escalation:

### GOLD_031
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I understand your frustration regarding the Christmas priority booking slots. The historical information available to me doesn't specify how a delivery free trial might affect eligibility for these slots. To look into this further for you, I'll need to escalate this to a human colleague who can access more detailed information about your account and the terms of your trial.

**Failure:** Rule did not escalate but human said YES

### GOLD_231
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I'm really sorry to hear about the two pints of milk leaking and ruining your car. That sounds incredibly frustrating. A human colleague will be able to look into this for you and log the details on our system. They may need your full name and address to help with this.

**Failure:** Rule did not escalate but human said YES

### GOLD_017
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> Hi there, I'm sorry to hear about this unexpected charge for your Delivery Saver Plan. So I can look into this for you, please send me a direct message with your full name, address, and email address, along with details of the charge.

**Failure:** Rule did not escalate but human said YES

### GOLD_243
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I understand your concerns regarding the handling of doughnuts, the use of napkins instead of tongs, and the potential cross-contamination risks for nut allergies in the display cabinet. I also see you're asking about a manager's response. As I don't have access to specific store operations, internal systems, or the details of your previous interaction, I'm unable to investigate this directly or comment on a manager's response. For a full investigation into these important matters, especially concerning allergy safety, please contact our customer service team directly. They will be able to look into this for you.

**Failure:** Rule did not escalate but human said YES

### GOLD_054
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I understand you're having trouble accessing the live chat feature on our website. I'm unable to check the real-time status of the live chat or resolve website functionality issues directly. For immediate assistance, please try contacting Tesco customer service through another channel or try accessing the live chat again later. A human colleague would be best equipped to provide an update on the live chat availability or assist with your query.

**Failure:** Rule did not escalate but human said YES

### GOLD_027
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> Hi there, I understand you're asking about Christmas delivery dates, especially for December 24th in London and Nottingham, and your booking for the 23rd. I don't have access to real-time delivery schedules for specific locations or individual bookings. For the most accurate information regarding your specific order and delivery date, please check your booking details or contact our customer service team directly. They will be able to confirm this for you.

**Failure:** Rule did not escalate but human said YES

### GOLD_247
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I understand you're looking to order an additional Clubcard for your husband, linked to your account. I don't have access to account details or the ability to arrange new cards. A human colleague will be able to look into this for you and explain the options for getting an additional card linked to your Clubcard account.

**Failure:** Rule did not escalate but human said YES

### GOLD_047
**Gold intent:** `N/A`  **Predicted:** `N/A`

**Generated reply:**
> I'm really sorry to hear that your pasta bake only had one day's date on it and you had to bin it so soon after purchase. That's definitely not the experience we want for our customers. To help us look into this, could you please provide the barcode, supplier code, and date code from the packaging, along with the store where you purchased it and your full name and postcode? A human colleague will be able to investigate this further for you.

**Failure:** Rule did not escalate but human said YES

## 9. Wrong Intent but Good Reply

These cases demonstrate that classification accuracy is not perfectly coupled to end-to-end response quality.

### GOLD_003
**Gold intent:** `delivery_issue`  **Predicted:** `store_service_complaint`

**Customer:**
> @Tesco estimate i've spent £78k with you but told today to pick my own shopping up from store means that figure will not be increasing!

**Generated reply:**
> I'm sorry to hear about your recent experience. So I can look into this for you, please could you DM your full name, full address, email address, which store you visited, and when this happened? Thank you.

**Failure:** Wrong intent but reply quality >= 4/5
**Judge reason:** The response directly addresses the customer's complaint by offering to investigate and appropriately requests necessary details, consistent with historical examples.

### GOLD_005
**Gold intent:** `online_website_issue`  **Predicted:** `pricing_offers`

**Customer:**
> So @Tesco as the Hudl 2 as with many android devices suffer from the WiFi security bug any plans to update it to patch the vulnerability?

**Generated reply:**
> Thanks for reaching out. Regarding software updates for the Hudl 2 to address WiFi security vulnerabilities, I don't have specific information on future update plans. This is a technical query that may require further investigation.

**Failure:** Wrong intent but reply quality >= 4/5
**Judge reason:** The response directly addresses the technical query, honestly states a lack of information, and appropriately suggests escalation for a specialized question.

### GOLD_009
**Gold intent:** `other`  **Predicted:** `store_service_complaint`

**Customer:**
> What an amazing time Year 4 had at Tesco this week learning all about Farm to Fork.

**Generated reply:**
> That's wonderful to hear! We're so glad Year 4 had an amazing time learning about Farm to Fork at Tesco.

**Failure:** Wrong intent but reply quality >= 4/5
**Judge reason:** The response perfectly acknowledges the customer's positive feedback in a polite and appropriate manner, aligning with historical examples.

### GOLD_015
**Gold intent:** `other`  **Predicted:** `product_quality_safety`

**Customer:**
> We have been down @Tesco today with the South Yorkshire Police Cadets collecting food #CAP #homelessness https://t.co/ENEKtgKhIw

**Generated reply:**
> That's wonderful to hear! Thank you and the South Yorkshire Police Cadets for collecting food to help with homelessness. We appreciate your efforts.

**Failure:** Wrong intent but reply quality >= 4/5
**Judge reason:** The response perfectly acknowledges and appreciates the customer's positive informational tweet, requiring no escalation.

### GOLD_016
**Gold intent:** `other`  **Predicted:** `product_quality_safety`

**Customer:**
> Who did this @Tesco https://t.co/zOiojg6aE3

**Generated reply:**
> Hi there! Thanks for reaching out. To help me understand what's happened and look into this for you, could you please send us a direct message with more details about the issue shown in the link? Please include your full name, address, and email if it's related to an order or store visit. Thanks!

**Failure:** Wrong intent but reply quality >= 4/5
**Judge reason:** The response appropriately asks for more details via DM to understand the vague customer query, aligning with historical examples and best practices.

### GOLD_021
**Gold intent:** `other`  **Predicted:** `product_quality_safety`

**Customer:**
> @Tesco perhaps you are naughty for having these early perhaps im worse for eating it but it was so good. https://t.co/1rTANOQ3vL

**Generated reply:**
> Hi there! We're so glad to hear you enjoyed it! Thanks for sharing your feedback. 😊

**Failure:** Wrong intent but reply quality >= 4/5
**Judge reason:** The response perfectly acknowledges the customer's positive feedback in a polite and appropriate manner, requiring no escalation.

### GOLD_017
**Gold intent:** `pricing_offers`  **Predicted:** `delivery_slots`

**Customer:**
> @Tesco why have I been charged a random £2.00 for the Delivery Saver Plan? 🤔

**Generated reply:**
> Hi there, I'm sorry to hear about this unexpected charge for your Delivery Saver Plan. So I can look into this for you, please send me a direct message with your full name, address, and email address, along with details of the charge.

**Failure:** Wrong intent but reply quality >= 4/5
**Judge reason:** The response correctly identifies the issue and provides the appropriate next step for escalation, aligning with historical support behavior for account-specific queries.

### GOLD_019
**Gold intent:** `product_information`  **Predicted:** `product_quality_safety`

**Customer:**
> Why can’t @tesco just stop putting reduced stickers over cooking instructions 😩 how long I’m I supposed to have this oven ???? https://t.co/eRxuRVC2t4

**Generated reply:**
> I'm sorry to hear that the reduced sticker covered the cooking instructions on your product. That sounds really frustrating! So I can look into this, could you please let me know which store you purchased this item from and what the product was? Thanks!

**Failure:** Wrong intent but reply quality >= 4/5
**Judge reason:** The response directly addresses the customer's complaint, shows empathy, and appropriately gathers necessary information for investigation, aligning with historical support practices.

## 10. Correct Intent but Bad Reply

These cases isolate failures in retrieval, generation, grounding, or escalation despite correct classification.

### GOLD_060
**Gold intent:** `clubcard_account`  **Predicted:** `clubcard_account`

**Customer:**
> @Tesco if I Exchange my club vouchers for pizza of cineworld how long do the exchanges vouchers last for ??

**Generated reply:**
> Hi there! Clubcard Boost tokens, including those for Pizza Express and Cineworld, are generally valid for 6 months from the day you exchange your vouchers for them. I hope this helps!

**Failure:** Correct intent but reply quality <= 3/5
**Judge reason:** The response directly answers the customer's question but makes a specific factual claim about the validity period that is not supported by the provided historical examples.

### GOLD_012
**Gold intent:** `pricing_offers`  **Predicted:** `pricing_offers`

**Customer:**
> But in @Tesco Apple Pay is still limited to £30. Such a shame. No limit at @Morrisons I’m pleased to say :-) https://t.co/7sb1KkGWvS

**Generated reply:**
> Thanks for sharing your feedback regarding the Apple Pay limit at Tesco. We appreciate you letting us know about your experience.

**Failure:** Correct intent but reply quality <= 3/5
**Judge reason:** The response politely acknowledges the customer's feedback but misses the opportunity to provide helpful information about the Apple Pay limit, which has been historically provided by Tesco.

## 11. Low Grounding

### GOLD_028
**Gold intent:** `product_availability`  **Predicted:** `online_order_issue`

**Customer:**
> hey @Tesco why no courgettes for online customers been trying for three weeks. Am now looking at @sainsburys and they have them may switch

**Generated reply:**
> Hi there! I'm sorry to hear you're having trouble finding courgettes for your online orders. To help me look into this for you, could you please send me a direct message with your full name, address, and email address? Thank you!

**Failure:** Low grounding score
**Judge reason:** The response fails to address the customer's general question about courgette availability for online customers, instead immediately asking for personal details for an investigation the bot cannot perform.

### GOLD_029
**Gold intent:** `other`  **Predicted:** `pricing_offers`

**Customer:**
> I just got told off for putting the Christmas radio station on at work #bahhumbug

**Generated reply:**
> I'm sorry to hear you were told off at work. So I can look into this for you, please could you send us a direct message with your full name, full address, and email address?

**Failure:** Low grounding score
**Judge reason:** The model completely misunderstands the customer's message, assuming a Tesco staff issue when the customer is venting about their own workplace. It inappropriately offers to investigate and requests personal details.

### GOLD_060
**Gold intent:** `clubcard_account`  **Predicted:** `clubcard_account`

**Customer:**
> @Tesco if I Exchange my club vouchers for pizza of cineworld how long do the exchanges vouchers last for ??

**Generated reply:**
> Hi there! Clubcard Boost tokens, including those for Pizza Express and Cineworld, are generally valid for 6 months from the day you exchange your vouchers for them. I hope this helps!

**Failure:** Low grounding score
**Judge reason:** The response directly answers the customer's question but makes a specific factual claim about the validity period that is not supported by the provided historical examples.

### GOLD_002
**Gold intent:** `other`  **Predicted:** `product_quality_safety`

**Customer:**
> Seriously @Tesco  since when have you started cutting crusts of children’s sandwiches. Can’t believe your encouraging this!!! #parentinghell https://t.co/bNhABXSdK6

**Generated reply:**
> Hi there, thanks for reaching out. We're sorry to hear about your concern regarding our children's sandwiches. To help us look into this for you, could you please let us know which specific product you're referring to and where it was purchased? We'll then pass this on to our team.

**Failure:** Low grounding score
**Judge reason:** The response misinterprets the customer's general policy question as a specific product complaint, asking for irrelevant details. While escalation for policy info is needed, the initial information gathering is misdirected.

### GOLD_012
**Gold intent:** `pricing_offers`  **Predicted:** `pricing_offers`

**Customer:**
> But in @Tesco Apple Pay is still limited to £30. Such a shame. No limit at @Morrisons I’m pleased to say :-) https://t.co/7sb1KkGWvS

**Generated reply:**
> Thanks for sharing your feedback regarding the Apple Pay limit at Tesco. We appreciate you letting us know about your experience.

**Failure:** Low grounding score
**Judge reason:** The response politely acknowledges the customer's feedback but misses the opportunity to provide helpful information about the Apple Pay limit, which has been historically provided by Tesco.

### GOLD_006
**Gold intent:** `other`  **Predicted:** `delivery_slots`

**Customer:**
> I do love  Christmas films!

**Generated reply:**
> It's lovely to hear you enjoy Christmas films! If you have any questions about your Tesco order or need assistance with anything else, please let me know.

**Failure:** Low grounding score
**Judge reason:** The response acknowledges the customer's statement but then immediately pivots to a generic support offer, missing the opportunity for conversational engagement shown in historical examples.

### GOLD_038
**Gold intent:** `product_information`  **Predicted:** `product_quality_safety`

**Customer:**
> @Tesco the cal content per sachet of your "easy oats, blueberry &amp; apple" is that before milk is added? or per made up sachet? its not clear!

**Generated reply:**
> Thanks for reaching out about the calorie content on our Easy Oats, Blueberry & Apple sachets. I understand it's not clear whether the calorie information is before or after milk is added, and I'm sorry for any confusion this causes. As an AI, I don't have access to real-time product details to clarify this for you directly. I can pass on your feedback about the labeling to our team for review. A human colleague would be best placed to investigate this further or provide a definitive answer. Would you like me to escalate this so they can look into it?

**Failure:** Low grounding score
**Judge reason:** The AI correctly identifies the customer's query but claims it cannot access product details, despite historical examples showing this information was directly provided by agents. This leads to unnecessary escalation for a static product detail.

### GOLD_212
**Gold intent:** `other`  **Predicted:** `product_information`

**Customer:**
> @Tesco is this a scam?? https://t.co/CQq0cK0Zui

**Generated reply:**
> Thanks for reaching out about this link. I can't directly confirm if it's a scam, but I understand why you'd be concerned. For issues like this, it's best for a human colleague to look into it and provide you with the correct guidance on how to report it.

**Failure:** Low grounding score
**Judge reason:** The response correctly identifies its limitation and escalates, but misses the opportunity to provide the reporting email address seen in historical examples.

## 12. Low Safety / Factuality

### GOLD_029
**Gold intent:** `other`  **Predicted:** `pricing_offers`

**Customer:**
> I just got told off for putting the Christmas radio station on at work #bahhumbug

**Generated reply:**
> I'm sorry to hear you were told off at work. So I can look into this for you, please could you send us a direct message with your full name, full address, and email address?

**Failure:** Low safety/factuality score
**Judge reason:** The model completely misunderstands the customer's message, assuming a Tesco staff issue when the customer is venting about their own workplace. It inappropriately offers to investigate and requests personal details.

### GOLD_060
**Gold intent:** `clubcard_account`  **Predicted:** `clubcard_account`

**Customer:**
> @Tesco if I Exchange my club vouchers for pizza of cineworld how long do the exchanges vouchers last for ??

**Generated reply:**
> Hi there! Clubcard Boost tokens, including those for Pizza Express and Cineworld, are generally valid for 6 months from the day you exchange your vouchers for them. I hope this helps!

**Failure:** Low safety/factuality score
**Judge reason:** The response directly answers the customer's question but makes a specific factual claim about the validity period that is not supported by the provided historical examples.

### GOLD_028
**Gold intent:** `product_availability`  **Predicted:** `online_order_issue`

**Customer:**
> hey @Tesco why no courgettes for online customers been trying for three weeks. Am now looking at @sainsburys and they have them may switch

**Generated reply:**
> Hi there! I'm sorry to hear you're having trouble finding courgettes for your online orders. To help me look into this for you, could you please send me a direct message with your full name, address, and email address? Thank you!

**Failure:** Low safety/factuality score
**Judge reason:** The response fails to address the customer's general question about courgette availability for online customers, instead immediately asking for personal details for an investigation the bot cannot perform.

### GOLD_038
**Gold intent:** `product_information`  **Predicted:** `product_quality_safety`

**Customer:**
> @Tesco the cal content per sachet of your "easy oats, blueberry &amp; apple" is that before milk is added? or per made up sachet? its not clear!

**Generated reply:**
> Thanks for reaching out about the calorie content on our Easy Oats, Blueberry & Apple sachets. I understand it's not clear whether the calorie information is before or after milk is added, and I'm sorry for any confusion this causes. As an AI, I don't have access to real-time product details to clarify this for you directly. I can pass on your feedback about the labeling to our team for review. A human colleague would be best placed to investigate this further or provide a definitive answer. Would you like me to escalate this so they can look into it?

**Failure:** Low safety/factuality score
**Judge reason:** The AI correctly identifies the customer's query but claims it cannot access product details, despite historical examples showing this information was directly provided by agents. This leads to unnecessary escalation for a static product detail.

### GOLD_005
**Gold intent:** `online_website_issue`  **Predicted:** `pricing_offers`

**Customer:**
> So @Tesco as the Hudl 2 as with many android devices suffer from the WiFi security bug any plans to update it to patch the vulnerability?

**Generated reply:**
> Thanks for reaching out. Regarding software updates for the Hudl 2 to address WiFi security vulnerabilities, I don't have specific information on future update plans. This is a technical query that may require further investigation.

**Failure:** Low safety/factuality score
**Judge reason:** The response directly addresses the technical query, honestly states a lack of information, and appropriately suggests escalation for a specialized question.

### GOLD_006
**Gold intent:** `other`  **Predicted:** `delivery_slots`

**Customer:**
> I do love  Christmas films!

**Generated reply:**
> It's lovely to hear you enjoy Christmas films! If you have any questions about your Tesco order or need assistance with anything else, please let me know.

**Failure:** Low safety/factuality score
**Judge reason:** The response acknowledges the customer's statement but then immediately pivots to a generic support offer, missing the opportunity for conversational engagement shown in historical examples.

### GOLD_001
**Gold intent:** `store_service_complaint`  **Predicted:** `store_service_complaint`

**Customer:**
> @Tesco why your man in Ewell tryna front me outside over a mistake on his behalf why every time I go there they are they so rude&amp;abusive?!

**Generated reply:**
> I'm very sorry to hear about your experience at our Ewell store. That's certainly not the standard of service we aim for. Could you please provide more details about what happened, including the date and time of your visit? This will help us look into the matter further.

**Failure:** Low safety/factuality score
**Judge reason:** The response directly addresses the serious complaint, expresses empathy, and appropriately requests necessary details for investigation, aligning well with historical examples and the need for human intervention.

### GOLD_002
**Gold intent:** `other`  **Predicted:** `product_quality_safety`

**Customer:**
> Seriously @Tesco  since when have you started cutting crusts of children’s sandwiches. Can’t believe your encouraging this!!! #parentinghell https://t.co/bNhABXSdK6

**Generated reply:**
> Hi there, thanks for reaching out. We're sorry to hear about your concern regarding our children's sandwiches. To help us look into this for you, could you please let us know which specific product you're referring to and where it was purchased? We'll then pass this on to our team.

**Failure:** Low safety/factuality score
**Judge reason:** The response misinterprets the customer's general policy question as a specific product complaint, asking for irrelevant details. While escalation for policy info is needed, the initial information gathering is misdirected.

## 13. Recommended Cases for Final Report

The final 6-page report should not include dozens of examples. A compact set of representative cases is more useful.

- **Classification:** `GOLD_028` — product_availability -> online_order_issue
- **Retrieval:** `GOLD_003` — Human retrieval score = 0.0
- **Escalation false positive:** `GOLD_029` — Gemini escalated but human said NO
- **Escalation false negative:** `GOLD_003` — Gemini did not escalate but human said YES
- **Unsupported claim:** `GOLD_028` — Judge flagged unsupported claim
- **Wrong intent but good reply:** `GOLD_003` — Wrong intent but reply quality >= 4/5
- **Correct intent but bad reply:** `GOLD_060` — Correct intent but reply quality <= 3/5

These examples should be manually reviewed before inclusion in the final report. Automated ranking is used only to surface candidates; it does not replace qualitative analysis.
