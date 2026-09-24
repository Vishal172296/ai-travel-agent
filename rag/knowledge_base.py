"""
Pre-loaded Travel Knowledge Base for AI Travel Agent RAG Engine.
Contains comprehensive destination guides, visa regulations, cultural etiquette,
local transit hacks, safety tips, packing advice, and culinary highlights.
"""

DESTINATIONS_DATA = [
    {
        "destination": "Dubai, United Arab Emirates",
        "keywords": ["dubai", "uae", "emirates", "burj khalifa", "dxb"],
        "category": "Destination Guide",
        "title": "Dubai Complete Travel & Visa Guide",
        "content": (
            "DESTINATION OVERVIEW: Dubai is a world-class cosmopolitan hub in the UAE known for luxury shopping, "
            "ultramodern architecture, desert safaris, and lively nightlife.\n\n"
            "VISA REQUIREMENTS:\n"
            "- Indian Passport Holders: Eligible for Visa on Arrival (14 days extendable) if holding a valid US Visa, "
            "US Green Card, UK Residence Visa, or EU Residence Visa with at least 6 months validity. Otherwise, a 30-day "
            "or 60-day tourist e-visa must be pre-applied via airlines (Emirates/Flydubai) or travel portals. Passport must "
            "be valid for at least 6 months from entry.\n"
            "- US/UK/EU/Australian Citizens: 30-day free visa-on-arrival stamped at immigration.\n\n"
            "BEST TIME TO VISIT & WEATHER:\n"
            "- Peak Season: November to March (Pleasant weather, 20°C to 28°C, ideal for outdoor sightseeing and desert safaris).\n"
            "- Shoulder Season: April & October (Warm, 30°C to 36°C, lower hotel rates).\n"
            "- Off-Peak Summer: May to September (Extreme heat 40°C-48°C, heavy discounts, focus on indoor malls & theme parks).\n\n"
            "TOP ATTRACTIONS & EXPERIENCES:\n"
            "1. Burj Khalifa & Dubai Mall: 124th/148th floor At The Top view, Dubai Fountain evening water show.\n"
            "2. Desert Safari: Dune bashing, camel rides, quad biking, BBQ dinner with cultural dance.\n"
            "3. Dubai Marina & Palm Jumeirah: Marina yacht cruise, Atlantis Aquaventure waterpark, The View at The Palm.\n"
            "4. Old Dubai (Deira & Bur Dubai): Gold Souk, Spice Souk, Abra traditional boat ride across Dubai Creek (AED 1).\n"
            "5. Museum of the Future: Cutting-edge futuristic architectural and interactive technology marvel.\n\n"
            "LOCAL TRANSPORTATION & HACKS:\n"
            "- Dubai Metro: Buy a Silver Nol Card (AED 25 with AED 19 credit) for easy travel across Red and Green lines.\n"
            "- Taxis & Ride Hailing: Careem and Uber are widely available. Official RTA Dubai taxis (metered) are cheaper.\n"
            "- Airport Transit: Dubai Metro Red Line directly connects Terminal 1 and Terminal 3 to Downtown.\n\n"
            "CULTURAL DOS & DON'TS:\n"
            "- DO dress modestly in religious, traditional, and government areas (cover shoulders and knees).\n"
            "- DO NOT drink alcohol in public spaces or streets; drinking is only permitted in licensed venues, hotels, and bars.\n"
            "- DO NOT photograph local women, government/military buildings without permission.\n"
            "- Public displays of affection (intense kissing/embracing) are culturally offensive and legally prohibited.\n\n"
            "EMERGENCY & USEFUL NUMBERS:\n"
            "- Police: 999 | Ambulance: 998 | Fire: 997 | Tourist Police: 901\n"
            "- Currency: UAE Dirham (AED), approx 1 USD = 3.67 AED, 1 AED = ~22.8 INR."
        ),
    },
    {
        "destination": "Bali, Indonesia",
        "keywords": ["bali", "indonesia", "ubud", "seminyak", "canggu", "dps"],
        "category": "Destination Guide",
        "title": "Bali Ultimate Island Travel Guide",
        "content": (
            "DESTINATION OVERVIEW: Bali is Indonesia's most celebrated island, renowned for volcanic mountains, iconic "
            "rice paddies, coral reefs, spiritual Hindu temples, and vibrant beach surf breaks.\n\n"
            "VISA REQUIREMENTS:\n"
            "- Visa on Arrival (VoA): Available for 90+ nationalities including India, US, UK, Australia, EU.\n"
            "- Cost: IDR 500,000 (~$35 USD) payable online (e-VoA) or at Bali airport on arrival. Valid for 30 days, "
            "extendable once for another 30 days.\n"
            "- Mandatory Tourist Levy: IDR 150,000 (~$10 USD) per tourist, payable via the official 'Love Bali' app.\n"
            "- Passport must have minimum 6 months validity and return flight ticket.\n\n"
            "BEST TIME TO VISIT:\n"
            "- Dry Season: April to October (Sunny skies, low humidity, best for surfing, beaches, and hiking Mount Batur).\n"
            "- Wet Season: November to March (Tropical afternoon downpours, lush greenery, lower hotel tariffs, fewer crowds).\n\n"
            "TOP ATTRACTIONS BY REGION:\n"
            "1. Ubud (Cultural Heart): Tegalalang Rice Terraces, Sacred Monkey Forest Sanctuary, Campuhan Ridge Walk, art markets.\n"
            "2. Canggu & Seminyak: Beach clubs (Finns, Potato Head), surf lessons, trendy vegan cafes, sunsets.\n"
            "3. Uluwatu: Cliffside Uluwatu Temple, famous sunset Kecak fire dance, Single Fin surf point.\n"
            "4. Nusa Penida Day Trip: Kelingking T-Rex Beach, Broken Beach, Angel's Billabong, snorkeling with Manta Rays.\n"
            "5. Mount Batur Sunrise Trek: 2-hour early morning hike to watch sunrise above the clouds.\n\n"
            "LOCAL TRANSPORT & TIPS:\n"
            "- Scooter Rental: IDR 70,000 - 100,000/day (~$5-7 USD). International Driving Permit (IDP) with motorcycle endorsement required.\n"
            "- Apps: Download 'Gojek' and 'Grab' for cheap motorbike taxis, car rides, and food delivery.\n"
            "- Temple Etiquette: Wear a sarong and sash (usually available for rent/free at entrances) covering knees and shoulders.\n"
            "- Tap Water: Do NOT drink tap water (risk of 'Bali Belly'). Drink only bottled or filtered water.\n\n"
            "EMERGENCY CONTACTS:\n"
            "- Police: 110 | Ambulance: 118 | Tourist Assistance: +62 361 754599\n"
            "- Currency: Indonesian Rupiah (IDR), approx 1 USD = ~15,800 IDR."
        ),
    },
    {
        "destination": "Paris, France",
        "keywords": ["paris", "france", "europe", "eiffel tower", "cdg", "orly", "schengen"],
        "category": "Destination Guide",
        "title": "Paris & Schengen Travel & Etiquette Guide",
        "content": (
            "DESTINATION OVERVIEW: Paris, the City of Light, is global epicenter for art, fashion, gastronomy, and "
            "timeless architecture nestled along the River Seine.\n\n"
            "VISA REQUIREMENTS:\n"
            "- Schengen Visa: Required for non-EU/non-EEA passport holders (including India, China, etc.). Apply at French VFS "
            "at least 3-6 weeks before travel. Requires travel insurance with minimum €30,000 coverage, flight itinerary, hotel "
            "vouchers, and proof of funds.\n"
            "- US/UK/Canada/Australia citizens: 90 days visa-free entry within any 180-day window.\n\n"
            "BEST TIME TO VISIT:\n"
            "- Spring (April-May) and Autumn (September-October): Mild weather, blossoming gardens, fewer tourist queues.\n"
            "- Summer (June-August): Long sunny daylight hours (until 10 PM), open-air cinema, but crowded and warmer.\n"
            "- Winter (November-February): Crisp air, festive Christmas markets, lowest hotel rates.\n\n"
            "TOP ATTRACTIONS & EXPERIENCES:\n"
            "1. Eiffel Tower: Book summit tickets 60 days in advance online to skip 2-hour lines. Sparkles every hour after sunset.\n"
            "2. Musée du Louvre: Home to Mona Lisa, Venus de Milo. Closed on Tuesdays. Pre-book timed entry ticket.\n"
            "3. Montmartre & Sacré-Cœur: Bohemian hilltop neighborhood with panoramic city views and street artists.\n"
            "4. Seine River Cruise: 1-hour evening cruise (Bateaux Parisiens / Vedettes du Pont Neuf) for illuminated monuments.\n"
            "5. Palace of Versailles: Extravagant royal palace & Hall of Mirrors, 45 minutes train ride on RER C.\n\n"
            "TRANSIT & LOCAL HACKS:\n"
            "- Paris Metro: The fastest way to get around. Buy a 'Navigo Easy' card or bundle of 10 digital T+ tickets (Carnet).\n"
            "- Airport Connection: RER B train connects CDG Airport directly to Châtelet-Les Halles / Gare du Nord (~€11.80).\n"
            "- Tipping: Service compris (service charge) is legally included in bills. Leaving 5-10% in coins for exceptional service is customary.\n"
            "- Scams to Avoid: Fake petition clipboard girls around Eiffel Tower/Louvre, friendship bracelet scammers at Sacré-Cœur steps, pickpockets on Metro Line 1.\n\n"
            "EMERGENCY NUMBERS:\n"
            "- European Emergency: 112 | Police: 17 | Ambulance (SAMU): 15\n"
            "- Currency: Euro (€), approx 1 EUR = ~1.08 USD, ~90 INR."
        ),
    },
    {
        "destination": "Tokyo, Japan",
        "keywords": ["tokyo", "japan", "asia", "hnd", "nrt", "shinjuku", "shibuya"],
        "category": "Destination Guide",
        "title": "Tokyo Modern & Traditional Explorer Guide",
        "content": (
            "DESTINATION OVERVIEW: Tokyo is a hyper-futuristic metropolis harmoniously blended with historic temples, "
            "michelin-starred culinary mastery, anime culture, and unparalleled safety.\n\n"
            "VISA REQUIREMENTS:\n"
            "- Indian Passports: eVisa available online for single-entry short-term tourism (up to 90 days) via eVisa portal.\n"
            "- US/UK/EU/Singapore/Australia: 90 days visa-free tourism.\n"
            "- Passport must be valid for the duration of stay.\n\n"
            "BEST TIME TO VISIT:\n"
            "- Cherry Blossom (Sakura): Late March to early April (breathtaking pink blossoms in Ueno and Shinjuku Gyoen).\n"
            "- Autumn Foliage: October to November (vibrant red maples, crisp clear 15-20°C weather).\n"
            "- Winter (December-February): Cold, sunny days, best visibility of Mount Fuji from Tokyo Skytree.\n\n"
            "MUST-VISIT DISTRICTS:\n"
            "1. Shibuya: World's busiest pedestrian crossing, Hachiko statue, Shibuya Sky panoramic rooftop observation deck.\n"
            "2. Shinjuku: Neon-lit skyscraper district, Omoide Yokocho (Memory Lane yakitori alleys), Kabukicho, Golden Gai.\n"
            "3. Asakusa: Senso-ji, Tokyo's oldest Buddhist temple, Nakamise shopping street for traditional matcha sweets.\n"
            "4. Akihabara: World center of manga, anime, gaming arcades, and high-tech electronic megastores.\n"
            "5. Ginza & Tsukiji: Luxury shopping and mouthwatering fresh sushi breakfast at Tsukiji Outer Market.\n\n"
            "TRANSPORT & PRACTICAL TIPS:\n"
            "- IC Cards: Welcome Suica or Pasmo card on iPhone Wallet / physical card for contactless taps on subway, buses, and 7-Eleven.\n"
            "- JR Pass: Individual ticket or regional Tokyo Subway 72-hour pass (1,500 JPY) is far more economical than National JR Pass for city exploration.\n"
            "- Etiquette: Never eat while walking; don't tip (tipping is considered insulting in Japan); keep voice low on trains; stand on left side of escalators.\n"
            "- Cash: While cards/IC cards are widely accepted, carry 5,000-10,000 JPY cash for small ramen vending machines and temple stalls.\n\n"
            "EMERGENCY:\n"
            "- Police: 110 | Ambulance & Fire: 119 | Japan Travel Hotline: 050-3816-2720\n"
            "- Currency: Japanese Yen (JPY), approx 1 USD = ~152 JPY, 1 INR = ~1.7 JPY."
        ),
    },
    {
        "destination": "Goa, India",
        "keywords": ["goa", "india", "beach", "baga", "calangute", "anjuna", "panaji", "goi", "gox"],
        "category": "Destination Guide",
        "title": "Goa Beach & Heritage Complete Guide",
        "content": (
            "DESTINATION OVERVIEW: India's favorite coastal paradise, known for Portuguese colonial heritage, pristine "
            "sandy beaches, spicy seafood, water sports, and laid-back susegad lifestyle.\n\n"
            "NORTH GOA vs SOUTH GOA:\n"
            "- North Goa (Baga, Calangute, Anjuna, Vagator): Lively nightlife, flea markets, beach shacks, water sports (parasailing, jet-ski), party clubs (Tito's, Thalassa).\n"
            "- South Goa (Palolem, Agonda, Colva, Benaulim): Peaceful, scenic, romantic, luxury resorts, pristine clean white sand beaches, dolphin cruises.\n\n"
            "BEST TIME TO VISIT:\n"
            "- November to February: Peak winter season, perfect sunny 28°C weather, Sunburn festival, Christmas & New Year celebrations.\n"
            "- June to September: Lush monsoon season, dramatic waterfalls (Dudhsagar Trek), spice plantations, quiet green charm.\n\n"
            "TOP EXPERIENCES:\n"
            "1. Dudhsagar Falls: Majestic 4-tier waterfall inside Mollem National Park, jeep safari through jungle.\n"
            "2. Fontainhas (Panjim): Latin Quarter featuring colorful Portuguese villas, bakeries, and heritage art cafes.\n"
            "3. Fort Aguada & Chapora Fort: 17th-century coastal forts with panoramic Arabian Sea views.\n"
            "4. Churches of Old Goa: Basilica of Bom Jesus (storing remains of St. Francis Xavier), Se Cathedral.\n"
            "5. Spice Plantation Tour: Sahakari Spice Farm in Ponda with authentic Goan buffet on banana leaves.\n\n"
            "TRAVEL HACKS:\n"
            "- Bike/Scooter Rental: Rent an Activa or motorcycle for ₹350 - ₹500/day. Helmet mandatory; carry valid driving license.\n"
            "- Taxi: Download 'GoaMiles' government-approved taxi booking app for fixed regulated fares.\n"
            "- Food: Must try Goan Fish Curry Thali, Prawn Balchão, Pork Vindaloo, Bebinca dessert, and local Feni.\n\n"
            "EMERGENCY:\n"
            "- Police: 112 | Tourist Police: 0832-2428251 | Ambulance: 108"
        ),
    },
    {
        "destination": "London, United Kingdom",
        "keywords": ["london", "uk", "britain", "england", "lhr", "lgw", "big ben"],
        "category": "Destination Guide",
        "title": "London Heritage & Urban Guide",
        "content": (
            "DESTINATION OVERVIEW: The historic capital of the United Kingdom, renowned for royal landmarks, world-class "
            "free museums, West End theatre shows, and multicultural culinary scenes.\n\n"
            "VISA REQUIREMENTS:\n"
            "- Standard Visitor Visa required for non-visa national travelers (such as India). Apply 3 months ahead; valid for 6 months.\n"
            "- US/EU/Canadian/Australian tourists: Visa-free entry for up to 6 months.\n\n"
            "BEST TIME TO VISIT:\n"
            "- May to September: Warmest months (18°C-25°C), long daylight (sun sets at 9:30 PM in June), vibrant royal parks.\n"
            "- December: Magical festive lights on Oxford Street, Hyde Park Winter Wonderland.\n\n"
            "HIGHLIGHTS:\n"
            "1. Westminster & Royal London: Big Ben, Houses of Parliament, Westminster Abbey, Buckingham Palace Changing of the Guard.\n"
            "2. World Class Free Museums: British Museum, Natural History Museum, Victoria and Albert (V&A), Science Museum, Tate Modern (free general entry).\n"
            "3. Tower of London & Tower Bridge: Historic fortress, Crown Jewels, glass walkway across Tower Bridge.\n"
            "4. West End Theatres: Book Lion King, Phantom of the Opera, or Wicked; check TKTS booth at Leicester Square for half-price same-day tickets.\n"
            "5. Borough Market: Iconic gourmet food market next to London Bridge serving artisanal cheese, salt beef bagels, and oysters.\n\n"
            "TRANSIT:\n"
            "- London Underground (The Tube): Tap in and out using contactless credit/debit card or Apple/Google Pay. Daily fare cap applies automatically.\n"
            "- Airport Transfer: Elizabeth Line or Piccadilly Line from Heathrow is much cheaper than the Heathrow Express.\n\n"
            "EMERGENCY:\n"
            "- Emergency (Police/Ambulance/Fire): 999 | Non-Emergency Police: 101\n"
            "- Currency: British Pound (£ GBP), 1 GBP = ~1.28 USD, ~107 INR."
        ),
    },
    {
        "destination": "Singapore",
        "keywords": ["singapore", "sin", "changi", "marina bay", "sentosa"],
        "category": "Destination Guide",
        "title": "Singapore Lion City Smart Travel Guide",
        "content": (
            "DESTINATION OVERVIEW: Singapore is a high-tech city-state known for architectural wonders, lush urban "
            "gardens, world-acclaimed street food in hawker centres, and extreme cleanliness.\n\n"
            "VISA REQUIREMENTS:\n"
            "- Indian Passports: Apply for an e-Visa through authorized visa agents. Processing takes 3-5 business days. Valid for 30-day stay.\n"
            "- US/UK/EU/Australian Passports: 90-day visa exemption upon arrival.\n"
            "- SG Arrival Card (SGAC): All foreign travelers must submit the free electronic SG Arrival Card with health declaration online within 3 days prior to arrival.\n\n"
            "BEST TIME TO VISIT:\n"
            "- Year-round destination (consistent tropical temperature 26°C-32°C). July-September has great outdoor events (Singapore Grand Prix F1 in September).\n\n"
            "TOP ATTRACTIONS:\n"
            "1. Gardens by the Bay: Futuristic Supertree Grove (free light show nightly at 7:45 & 8:45 PM), Cloud Forest waterfall dome.\n"
            "2. Marina Bay Sands SkyPark: Panoramic observation deck and iconic infinity pool overlooking Marina Bay.\n"
            "3. Sentosa Island: Universal Studios Singapore, S.E.A. Aquarium, Siloso Beach, cable car ride.\n"
            "4. Hawker Food Exploration: Maxwell Food Centre (Tian Tian Hainanese Chicken Rice), Lau Pa Sat (outdoor evening Satay street).\n"
            "5. Jewel Changi Airport: The Rain Vortex (world's tallest indoor waterfall), canopy park, and luxury transit shopping.\n\n"
            "LAWS & FINES (MUST KNOW):\n"
            "- Chewing gum is banned from sale and import.\n"
            "- Littering or jaywalking carries heavy on-the-spot fines starting from SGD $300.\n"
            "- Smoking is only permitted in designated yellow smoking boxes.\n\n"
            "TRANSIT:\n"
            "- MRT (Mass Rapid Transit): Spotless, ultra-efficient. Tap contactless Mastercard/Visa card at fare gates.\n"
            "- Taxis: Grab app works seamlessly everywhere.\n\n"
            "EMERGENCY:\n"
            "- Police: 999 | Ambulance: 995 | Tourist Hotline: 1800 736 2000\n"
            "- Currency: Singapore Dollar (SGD), 1 SGD = ~0.75 USD, ~63 INR."
        ),
    },
    {
        "destination": "General Travel Wisdom & Baggage Policies",
        "keywords": ["packing", "baggage", "luggage", "tips", "customs", "medicines", "electronics", "rules"],
        "category": "General Guide",
        "title": "Smart Packing, Baggage Rules & International Travel Hacks",
        "content": (
            "SMART PACKING ESSENTIALS:\n"
            "- Travel Documents: Physical and digital photocopies of Passport, Visa approval, Travel Insurance, Flight Tickets, and Hotel Vouchers.\n"
            "- Electronics & Charging: Universal international travel adapter with USB-C PD fast ports. Power banks (up to 20,000 mAh allowed ONLY in cabin hand luggage, NEVER in check-in baggage).\n"
            "- First-Aid & Medicines: Paracetamol, anti-diarrhea (Loperamide/ORS), motion sickness tablets, antihistamines, band-aids. Keep prescription drugs in original packaging with doctor's prescription note.\n"
            "- Money & Cards: Carry at least 2 zero-forex markup debit/credit cards (Visa/Mastercard) and approx $150-$200 USD equivalent in local cash for small merchants.\n\n"
            "AIRLINE CABIN & CHECK-IN BAGGAGE RULES:\n"
            "- Carry-on / Cabin Bag: Usually 7 kg max (55 x 35 x 25 cm) + 1 small personal item (laptop bag/handbag).\n"
            "- Liquids Rule (LAGs): Cabin liquids must be in containers of maximum 100 ml (3.4 oz) each, placed inside a single clear resealable 1-liter plastic bag.\n"
            "- Prohibited in Cabin: Scissors, pocket knives, corkscrews, lighters, liquids over 100ml.\n"
            "- Prohibited in Check-in: Spare lithium-ion batteries, power banks, e-cigarettes, valuable jewelry.\n\n"
            "AIRPORT TRANSIT & IMMIGRATION HACKS:\n"
            "- Arrive at airport 3 hours before international flights, 2 hours before domestic flights.\n"
            "- Fill out online arrival/customs cards in advance if the destination country offers an e-service.\n"
            "- Always keep your accommodation address and contact phone number readily accessible for immigration officers."
        ),
    },
]
