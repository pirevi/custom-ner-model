train_data = [
    ("Sarah Johnson can be reached at sarah.johnson@example.com.", {"entities": [(0, 13, "PERSON"), (32, 56, "EMAIL")]}),
    ("Please contact James Brown at +44 7911 123456.", {"entities": [(15, 26, "PERSON"), (30, 43, "CONTACT_NUMBER")]}),
    ("Emily Davis visited the Tesco store in Manchester.", {"entities": [(0, 11, "PERSON"), (38, 48, "GPE")]}),
    ("Michael Wilson's Tesco Clubcard number is 1234-5678-9012-3456.", {"entities": [(0, 15, "PERSON"), (44, 63, "CLUBCARD_NUMBER")]}),
    ("Lisa Taylor can be emailed at lisa.taylor@tesco.com.", {"entities": [(0, 11, "PERSON"), (30, 48, "EMAIL")]}),
    ("David Evans called from 020 7946 0958.", {"entities": [(0, 11, "PERSON"), (25, 38, "CONTACT_NUMBER")]}),
    ("Rachel Green's Tesco Clubcard is 9876-5432-1098-7654.", {"entities": [(0, 12, "PERSON"), (37, 56, "CLUBCARD_NUMBER")]}),
    ("Mark Harris lives in Leeds.", {"entities": [(0, 10, "PERSON"), (21, 26, "GPE")]}),
    ("Olivia White can be reached at olivia.white@example.co.uk.", {"entities": [(0, 12, "PERSON"), (31, 54, "EMAIL")]}),
    ("Daniel Clark's phone number is 0113 496 7890.", {"entities": [(0, 12, "PERSON"), (31, 44, "CONTACT_NUMBER")]}),
    ("John Smith used his Tesco Clubcard 4567-8901-2345-6789 in Birmingham.", {"entities": [(0, 10, "PERSON"), (31, 50, "CLUBCARD_NUMBER"), (54, 64, "GPE")]}),
    ("Emma Watson's email is emma.watson@tesco.com.", {"entities": [(0, 11, "PERSON"), (24, 43, "EMAIL")]}),
    ("Robert Brown called from +44 7800 123456.", {"entities": [(0, 12, "PERSON"), (26, 39, "CONTACT_NUMBER")]}),
    ("Sophia Lee's Tesco Clubcard number is 3210-9876-5432-1098.", {"entities": [(0, 10, "PERSON"), (41, 60, "CLUBCARD_NUMBER")]}),
    ("William Taylor visited the Tesco store in Bristol.", {"entities": [(0, 14, "PERSON"), (41, 48, "GPE")]}),
    ("Charlotte Harris can be contacted at charlotte.harris@example.com.", {"entities": [(0, 16, "PERSON"), (39, 64, "EMAIL")]}),
    ("George Martin's phone number is 0161 496 1234.", {"entities": [(0, 13, "PERSON"), (32, 45, "CONTACT_NUMBER")]}),
    ("Amelia Wilson used her Tesco Clubcard 7890-1234-5678-9012.", {"entities": [(0, 13, "PERSON"), (34, 53, "CLUBCARD_NUMBER")]}),
    ("Noah Brown lives in Sheffield.", {"entities": [(0, 10, "PERSON"), (21, 30, "GPE")]}),
    ("Mia Johnson's email is mia.johnson@tesco.com.", {"entities": [(0, 12, "PERSON"), (25, 44, "EMAIL")]})
]

test_data = [
    ("Anna Smith can be reached at anna.smith@example.com.", {"entities": [(0, 10, "PERSON"), (29, 50, "EMAIL")]}),
    ("Please contact Laura Brown at +44 7911 654321.", {"entities": [(15, 26, "PERSON"), (30, 43, "CONTACT_NUMBER")]}),
    ("Sophie Davis visited the Tesco store in London.", {"entities": [(0, 12, "PERSON"), (39, 45, "GPE")]}),
    ("Oliver Wilson's Tesco Clubcard number is 2345-6789-0123-4567.", {"entities": [(0, 13, "PERSON"), (44, 63, "CLUBCARD_NUMBER")]}),
    ("Ella Taylor can be emailed at ella.taylor@tesco.com.", {"entities": [(0, 10, "PERSON"), (29, 48, "EMAIL")]}),
    ("Liam Evans called from 020 7946 1234.", {"entities": [(0, 10, "PERSON"), (24, 37, "CONTACT_NUMBER")]}),
    ("Grace Green's Tesco Clubcard is 8765-4321-0987-6543.", {"entities": [(0, 11, "PERSON"), (36, 55, "CLUBCARD_NUMBER")]}),
    ("Harry Harris lives in Glasgow.", {"entities": [(0, 11, "PERSON"), (22, 29, "GPE")]}),
    ("Isla White can be reached at isla.white@example.co.uk.", {"entities": [(0, 10, "PERSON"), (29, 52, "EMAIL")]}),
    ("Jack Clark's phone number is 0113 496 0123.", {"entities": [(0, 10, "PERSON"), (29, 42, "CONTACT_NUMBER")]}),
    ("Ethan Smith used his Tesco Clubcard 5678-9012-3456-7890 in Liverpool.", {"entities": [(0, 11, "PERSON"), (32, 51, "CLUBCARD_NUMBER"), (55, 64, "GPE")]}),
    ("Chloe Watson's email is chloe.watson@tesco.com.", {"entities": [(0, 12, "PERSON"), (25, 44, "EMAIL")]}),
    ("James Brown called from +44 7800 654321.", {"entities": [(0, 11, "PERSON"), (25, 38, "CONTACT_NUMBER")]}),
    ("Ruby Lee's Tesco Clubcard number is 4321-0987-6543-2109.", {"entities": [(0, 8, "PERSON"), (39, 58, "CLUBCARD_NUMBER")]}),
    ("William Taylor visited the Tesco store in Edinburgh.", {"entities": [(0, 14, "PERSON"), (41, 50, "GPE")]}),
    ("Emily Harris can be contacted at emily.harris@example.com.", {"entities": [(0, 12, "PERSON"), (35, 60, "EMAIL")]}),
    ("George Martin's phone number is 0161 496 5678.", {"entities": [(0, 13, "PERSON"), (32, 45, "CONTACT_NUMBER")]}),
    ("Amelia Wilson used her Tesco Clubcard 8901-2345-6789-0123.", {"entities": [(0, 13, "PERSON"), (34, 53, "CLUBCARD_NUMBER")]}),
    ("Noah Brown lives in Nottingham.", {"entities": [(0, 10, "PERSON"), (21, 32, "GPE")]}),
    ("Mia Johnson's email is mia.johnson@tesco.com.", {"entities": [(0, 12, "PERSON"), (25, 44, "EMAIL")]})
]