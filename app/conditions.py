"""Plain-language facts shown on the result card. General education only, not advice."""

DISPLAY_NAMES = {
    "Contact Dermatitis": "Contact Dermatitis",
    "Eczema": "Eczema (atopic dermatitis)",
    "Healthy": "Healthy skin",
    "Psoriasis": "Psoriasis",
    "Scabies": "Scabies",
    "Tinea Corporis": "Tinea Corporis (ringworm)",
}

CONDITIONS = {
    "Contact Dermatitis": {
        "summary": "An itchy, red rash caused by touching something the skin reacts to, such as nickel, "
                   "cosmetics, detergents, plants or rubber.",
        "typical_sites": "Where the trigger touched the skin: hands, wrists, face, neck, feet.",
        "next_steps": "Avoid the suspected trigger and see a dermatologist, especially if the rash spreads, "
                      "blisters or does not settle within two weeks.",
    },
    "Eczema": {
        "summary": "A long-lasting condition that makes skin dry, itchy and inflamed, often coming in flares. "
                   "It is common in children and people with allergies or asthma.",
        "typical_sites": "Folds of elbows and knees, hands, neck, face (in infants: cheeks and scalp).",
        "next_steps": "Regular moisturising and avoiding harsh soaps help; a dermatologist can advise on "
                      "treatment for flares.",
    },
    "Psoriasis": {
        "summary": "An immune-related condition in which skin cells build up quickly, forming raised, red or "
                   "dark patches covered with silvery scales. It is not contagious.",
        "typical_sites": "Elbows, knees, scalp, lower back, nails.",
        "next_steps": "See a dermatologist for confirmation; effective long-term treatments exist.",
    },
    "Scabies": {
        "summary": "An intensely itchy rash caused by tiny mites that burrow into the skin. It spreads by close "
                   "skin contact and often affects several people in a household.",
        "typical_sites": "Between fingers, wrists, elbows, waist, buttocks, groin; itch is worse at night.",
        "next_steps": "See a doctor promptly - the whole household is usually treated at the same time.",
    },
    "Tinea Corporis": {
        "summary": "A fungal skin infection (ringworm) that forms ring-shaped, scaly, itchy patches with a "
                   "clearer centre. It spreads by contact with people, pets or shared items.",
        "typical_sites": "Trunk, arms, legs, groin (tinea cruris), face.",
        "next_steps": "See a doctor for the right antifungal treatment; avoid steroid creams without "
                      "advice, which can make it worse.",
    },
    "Healthy": {
        "summary": "No sign of the five conditions DermaLens recognises was found in this photo.",
        "typical_sites": "-",
        "next_steps": "If you still have symptoms such as itching, pain or a changing mole, please see a "
                      "doctor - DermaLens only checks for five conditions.",
    },
}
