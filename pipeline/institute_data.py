"""
Curated institute data for CSAB/JoSAA predictor.
  - INSTITUTE_DOMAINS : exact domain lookup by institute name substring → official domain
  - GFTI_PERKS        : institute name → list of highlight strings shown as badges
"""

# ---------------------------------------------------------------------------
# Domain lookup  (substring match, checked longest-first inside the app)
# Each key is a unique enough substring of the full institute name.
# ---------------------------------------------------------------------------
INSTITUTE_DOMAINS: dict[str, str] = {
    # ── NITs ──────────────────────────────────────────────────────────────
    "National Institute of Technology, Tiruchirappalli":    "nitt.edu",
    "National Institute of Technology Karnataka":            "nitk.ac.in",
    "National Institute of Technology, Warangal":            "nitw.ac.in",
    "Visvesvaraya National Institute of Technology":         "vnit.ac.in",
    "Malaviya National Institute of Technology":             "mnit.ac.in",
    "Maulana Azad National Institute of Technology":         "manit.ac.in",
    "Motilal Nehru National Institute of Technology":        "mnnit.ac.in",
    "Sardar Vallabhbhai National Institute of Technology":   "svnit.ac.in",
    "Dr. B R Ambedkar National Institute of Technology":     "nitj.ac.in",
    "National Institute of Technology, Rourkela":            "nitrkl.ac.in",
    "National Institute of Technology Calicut":              "nitc.ac.in",
    "National Institute of Technology Durgapur":             "nitdgp.ac.in",
    "National Institute of Technology, Kurukshetra":         "nitkkr.ac.in",
    "National Institute of Technology, Silchar":             "nits.ac.in",
    "National Institute of Technology, Jamshedpur":          "nitjsr.ac.in",
    "National Institute of Technology Patna":                "nitp.ac.in",
    "National Institute of Technology Raipur":               "nitrr.ac.in",
    "National Institute of Technology Agartala":             "nita.ac.in",
    "National Institute of Technology Hamirpur":             "nith.ac.in",
    "National Institute of Technology Goa":                  "nitgoa.ac.in",
    "National Institute of Technology Delhi":                "nitdelhi.ac.in",
    "National Institute of Technology, Andhra Pradesh":      "nitandhra.ac.in",
    "National Institute of Technology Meghalaya":            "nitm.ac.in",
    "National Institute of Technology, Manipur":             "nitmanipur.ac.in",
    "National Institute of Technology, Mizoram":             "nitmz.ac.in",
    "National Institute of Technology Nagaland":             "nitnagaland.ac.in",
    "National Institute of Technology Puducherry":           "nitpy.ac.in",
    "National Institute of Technology Sikkim":               "nitsikkim.ac.in",
    "National Institute of Technology Arunachal Pradesh":    "nitap.ac.in",
    "National Institute of Technology, Srinagar":            "nitsri.ac.in",
    "National Institute of Technology, Uttarakhand":         "nituk.ac.in",
    "North Eastern Regional Institute of Science":           "nerist.ac.in",

    # ── IIITs ─────────────────────────────────────────────────────────────
    "Indian Institute of Information Technology, Allahabad": "iiita.ac.in",
    "Atal Bihari Vajpayee Indian Institute of Information Technology & Management Gwalior": "iiitm.ac.in",
    "Pt. Dwarka Prasad Mishra Indian Institute of Information Technology, Design & Manufacture Jabalpur": "iiitdmj.ac.in",
    "Indian Institute of Information Technology, Design & Manufacturing, Kancheepuram": "iiitdm.ac.in",
    "Indian Institute of Information Technology Design & Manufacturing Kurnool": "iiitk.ac.in",
    "Indian Institute of Information Technology (IIIT) Nagpur": "iiitn.ac.in",
    "Indian Institute of Information Technology (IIIT) Pune":   "iiitp.ac.in",
    "Indian Institute of Information Technology (IIIT) Ranchi": "iiitranchi.ac.in",
    "Indian Institute of Information Technology (IIIT), Sri City": "iiits.ac.in",
    "Indian Institute of Information Technology (IIIT)Kota":    "iiitkota.ac.in",
    "Indian Institute of Information Technology Bhagalpur":      "iiitbh.ac.in",
    "Indian Institute of Information Technology Bhopal":         "iiitbhopal.ac.in",
    "Indian Institute of Information Technology Guwahati":       "iiitg.ac.in",
    "Indian Institute of Information Technology Lucknow":        "iiitl.ac.in",
    "Indian Institute of Information Technology Manipur":        "iiitmanipur.ac.in",
    "Indian Institute of Information Technology Surat":          "iiitvadodara.ac.in",
    "Indian Institute of Information Technology Tiruchirappalli":"iiittirunelveli.ac.in",
    "Indian Institute of Information Technology(IIIT) Dharwad":  "iiitdwd.ac.in",
    "Indian Institute of Information Technology(IIIT) Kalyani":  "iiitkalyani.ac.in",
    "Indian Institute of Information Technology(IIIT) Kilohrad": "iiitkuk.ac.in",
    "Indian Institute of Information Technology(IIIT) Kottayam": "iiitkottayam.ac.in",
    "Indian Institute of Information Technology(IIIT) Una":      "iiitu.ac.in",
    "Indian Institute of Information Technology(IIIT), Vadodara, Gujrat": "iiitvadodara.ac.in",
    "Indian Institute of Information Technology, Agartala":      "iiitagartala.ac.in",
    "Indian Institute of Information Technology, Vadodara International Campus Diu": "iiitvadodara.ac.in",
    "International Institute of Information Technology, Bhubaneswar": "iiit-bh.ac.in",
    "International Institute of Information Technology, Naya Raipur":  "iiitnr.ac.in",
    "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY SENAPATI MANIPUR":     "iiitmanipur.ac.in",
    "Indian institute of information technology, Raichur":              "iiitbh.ac.in",

    # ── GFTIs ─────────────────────────────────────────────────────────────
    "Indian Institute of Engineering Science and Technology, Shibpur":  "iiest.ac.in", # NIRF 2025: 54
    "Delhi Technological University":                                    "dtu.ac.in", # NIRF 2025: 30
    "Netaji Subhas University of Technology":                            "nsut.ac.in", # NIRF 2025: 70
    "Punjab Engineering College":                                        "pec.ac.in", # NIRF 2025: 101-150 Band
    "Birla Institute of Technology, Mesra":                              "bitmesra.ac.in", # NIRF 2025: 51
    "Birla Institute of Technology, Deoghar":                            "bitmesra.ac.in",
    "Birla Institute of Technology, Patna":                              "bitmesra.ac.in",
    "Indira Gandhi Delhi Technical University for Women":                "igdtuw.ac.in", # NIRF 2025: Participating
    "Institute of Chemical Technology, Mumbai":                          "ictmumbai.edu.in", # NIRF 2025: 41
    "National Institute of Advanced Manufacturing Technology":           "niamt.ac.in", # NIRF 2025: Participating
    "National Institute of Foundry & Forge Technology":                  "nifft.ac.in",
    "Sant Longowal Institute of Engineering and Technology":             "sliet.ac.in", # NIRF 2025 Engineering: 79
    "Shri Mata Vaishno Devi University":                                 "smvdu.ac.in", # NIRF 2025 Engineering Band: 151-200
    "School of Planning & Architecture, New Delhi":                      "spa.ac.in", # NIRF 2025 Architecture: 8
    "School of Planning & Architecture, Bhopal":                         "spabhopal.ac.in", # NIRF 2025 Architecture: 11
    "School of Planning & Architecture: Vijayawada":                     "spaworld.org", # NIRF 2025 Architecture: 19
    "Assam University, Silchar":                                         "aus.ac.in",
    "Pondicherry Engineering College":                                   "pec.edu", # NIRF 2025: Participating
    "Puducherry Technological University":                                "ptuniv.ac.in", # NIRF 2025: Participating
    "Pandit Deendayal Energy University":                                "pdpu.ac.in", # NIRF 2025 Engineering: 98
    "Rajiv Gandhi National Aviation University":                         "rgnau.ac.in",
    "Manipal Institute of Technology":                                   "manipal.edu", # NIRF 2025 Overall (MAHE): 14
    "Islamic University of Science and Technology Kashmir":              "iust.ac.in",
    "Shri G. S. Institute of Technology and Science Indore":             "sgsits.ac.in",
    "Ghani Khan Choudhary Institute of Engineering and Technology":      "gkciet.ac.in",
    "Central institute of Technology Kokrajar":                          "cit.ac.in",
    "Chhattisgarh Swami Vivekanada Technical University":                "csvtu.ac.in", # NIRF 2025: Participating
    "Gautam Buddha University":                                          "gbu.ac.in",
    "Gati Shakti Vishwavidyalaya":                                       "gsv.ac.in",
    "National Institute of Electronics and Information Technology, Ajmer":      "nielit.gov.in",
    "National Institute of Electronics and Information Technology, Aurangabad":  "nielit.gov.in",
    "National Institute of Electronics and Information Technology, Gorakhpur":   "nielit.gov.in",
    "National Institute of Electronics and Information Technology, Patna":       "nielit.gov.in",
    "National Institute of Electronics and Information Technology, Ropar":       "nielit.gov.in",
    "National Institute of Food Technology Entrepreneurship and Management, Kundli": "niftem.ac.in",
    "National Institute of Food Technology Entrepreneurship and Management, Sonepat": "niftem.ac.in",
    "National Institute of Food Technology Entrepreneurship and Management, Thanjavur": "niftem-t.ac.in",
    "National Institute of Food Technology, Entrepreneurship and Management (NIFTEM) - Thanjavur": "niftem-t.ac.in",
    "Indian Institute of Carpet Technology":                             "iict.ac.in",
    "Indian Institute of Handloom Technology(IIHT), Varanasi":          "iiht.ac.in",
    "Indian Institute of Handloom Technology, Salem":                    "iiht.ac.in",
    "Indian Maritime University - Visakhapatnam":                        "imu.edu.in",
    "Institute of Infrastructure, Technology, Research and Management":  "iitram.ac.in", # NIRF 2025 State Univ Band: 51-100
    "Institute of Engineering and Technology, Dr. H. S. Gour University": "dhsgsu.ac.in",
    "Institute of Technology, Guru Ghasidas Vishwavidyalaya":           "ggu.ac.in",
    "School of Studies of Engineering and Technology, Guru Ghasidas":   "ggu.ac.in",
    "lndian Institute of Food Processing Technology":                    "iifpt.edu.in",
    "J.K. Institute of Applied Physics & Technology":                   "allduniv.ac.in",
    # Central Universities
    "Jawaharlal Nehru University":                                       "jnu.ac.in", # NIRF 2025 University: 2
    "University of Hyderabad":                                           "uohyd.ac.in", # NIRF 2025 Engineering: 74
    "Tezpur University":                                                 "tezu.ernet.in", # NIRF 2025 University: 78
    "Mizoram University":                                                "mzu.edu.in", # NIRF 2025 University: 81
    "Assam University":                                                  "aus.ac.in", # NIRF 2025 University: 97
    "HNB Garhwal University":                                            "hnbgu.ac.in",
    "Gurukula Kangri Vishwavidyalaya":                                   "gkv.ac.in",
    "North-Eastern Hill University":                                     "nehu.ac.in", # NIRF 2025 University Band: 151-200
    "CU Jharkhand":                                                      "cuj.ac.in",
    "Central University of Haryana":                                     "cuh.ac.in",
    "Central University of Jammu":                                       "cujammu.ac.in",
    "Central University of Rajasthan":                                   "curaj.ac.in",
}


# ---------------------------------------------------------------------------
# GFTI perks – shown as badge chips inside each institute card
# Keys are substrings of the institute name.
# ---------------------------------------------------------------------------
GFTI_PERKS: dict[str, list[str]] = {
    # ── Prestigious & well-known GFTIs ────────────────────────────────────
    "Indian Institute of Engineering Science and Technology, Shibpur": [
        "🏛️ 165+ year legacy (est. 1856)",
        "🔬 Oldest technical institute in Asia",
        "🎓 Deemed University status",
        "🤝 Strong industry & research ties",
        "🏆 NIRF 2025 Engineering Rank: 54",
    ],
    "Delhi Technological University": [
        "🏙️ Located in Delhi – metro advantage",
        "🎓 State University with strong alumni",
        "💼 Excellent campus placement record",
        "🤝 Collaborations with IITs & NITs",
        "🏆 NIRF 2025 Engineering Rank: 30",
    ],
    "Netaji Subhas University of Technology": [
        "🏙️ Located in Delhi",
        "🎓 State University",
        "🏆 NIRF 2025 Engineering Rank: 70",
        "💼 Strong Delhi NCR placements",
    ],
    "Punjab Engineering College": [
        "🏙️ Located in Chandigarh – UT advantage",
        "🎓 Deemed University status",
        "🏆 One of India's oldest engineering colleges (est. 1947)",
        "💼 Strong PSU & core sector placements",
        "📈 NIRF 2025 Engineering Rank Band: 101-150",
    ],
    "Birla Institute of Technology, Mesra": [
        "🎓 Deemed University – Ranchi",
        "🔬 Strong research culture",
        "🌏 Multiple off-campus centres across India",
        "💼 Good placements in IT & core engineering",
        "🏆 NIRF 2025 Engineering Rank: 51",
    ],
    "Birla Institute of Technology, Deoghar": [
        "🎓 BIT Mesra off-campus – Deemed Univ.",
        "💼 Access to BIT Mesra placement cell",
    ],
    "Birla Institute of Technology, Patna": [
        "🎓 BIT Mesra off-campus – Deemed Univ.",
        "🏙️ Located in Patna",
    ],
    "Indira Gandhi Delhi Technical University for Women": [
        "👩‍💻 India's first women-only engineering university",
        "🏙️ Located in Delhi",
        "💼 Strong Delhi NCR industry connections",
        "📈 NIRF 2024 Engineering Rank Band: 201-300",
    ],
    "Institute of Chemical Technology, Mumbai": [
        "🏙️ Located in Mumbai – financial capital",
        "🔬 Premier chemical & pharma research hub",
        "🏆 NIRF 2025 Engineering Rank: 41",
        "💼 Exceptional chemical & pharma sector placements",
        "🌏 Odisha campus collaboration",
    ],
    "Sant Longowal Institute of Engineering and Technology": [
        "🏛️ Central Government institute",
        "💰 Low tuition fees (CFTI status)",
        "🎓 Deemed University",
        "🏆 NIRF 2024 Engineering Rank: 76",
    ],
    "Shri Mata Vaishno Devi University": [
        "🏔️ Scenic Katra, J&K campus",
        "💰 Low fee structure",
        "🏛️ Central University status",
        "📈 NIRF 2024 Engineering Rank Band: 151-200",
    ],
    "School of Planning & Architecture, New Delhi": [
        "🏙️ Premier architecture school – New Delhi",
        "🏛️ One of only 3 SPAs funded by Govt. of India",
        "🎓 Deemed University",
        "🏆 Highly coveted for architecture & planning",
    ],
    "School of Planning & Architecture, Bhopal": [
        "🏛️ Government-funded SPA",
        "🏙️ Located in Bhopal",
        "🎓 Deemed University",
    ],
    "School of Planning & Architecture: Vijayawada": [
        "🏛️ Government-funded SPA – AP",
        "🎓 Deemed University",
    ],
    "National Institute of Advanced Manufacturing Technology": [
        "🏭 Specialised manufacturing technology focus",
        "🏛️ Ministry of Heavy Industries institute",
        "🔬 Research-oriented with industry tie-ups",
        "📍 Located in Ranchi",
    ],
    "National Institute of Foundry & Forge Technology": [
        "🏭 Unique foundry & forge engineering specialisation",
        "🏛️ Ministry of Education – govt. funded",
        "📍 Located in Ranchi",
        "💼 Niche but strong core sector placements",
    ],
    "Pandit Deendayal Energy University": [
        "⚡ Premier energy-sector engineering university",
        "📍 Located in Gandhinagar, Gujarat",
        "🤝 Strong oil & energy sector industry ties",
        "💼 Excellent placements in ONGC, GSPCL, Torrent",
    ],
    "Manipal Institute of Technology": [
        "🌏 International collaborations & exchange programs",
        "💼 Top IT & tech placements",
        "🎓 Deemed University – NAAC A++ accredited",
        "🏆 NIRF 2024 Engineering Rank: 56",
        "🏝️ Scenic Manipal campus",
    ],
    "Rajiv Gandhi National Aviation University": [
        "✈️ India's only aviation-focused national university",
        "🏛️ Ministry of Civil Aviation institute",
        "💼 Direct connections to aviation sector",
        "📍 Located in Fursatganj, UP",
    ],
    "Pondicherry Engineering College": [
        "📍 Located in Puducherry – UT advantage",
        "💰 Low fee structure",
        "🎓 Affiliated to Pondicherry University",
    ],
    "Puducherry Technological University": [
        "📍 Located in Puducherry – UT advantage",
        "💰 Low fee structure",
        "📈 NIRF 2024 Engineering Rank Band: 201-300",
    ],
    "Assam University, Silchar": [
        "🏛️ Central University",
        "📍 Located in Silchar, Assam",
        "💰 Affordable fee structure",
    ],
    "Islamic University of Science and Technology Kashmir": [
        "📍 Located in Kashmir",
        "🏛️ State University – J&K",
        "💰 Low fee structure",
    ],
    "Shri G. S. Institute of Technology and Science Indore": [
        "📍 Located in Indore – India's cleanest city",
        "🎓 Autonomous institute – RGPV affiliated",
        "💼 Strong Madhya Pradesh industry connections",
    ],
    "Ghani Khan Choudhary Institute of Engineering and Technology": [
        "📍 Located in Malda, West Bengal",
        "🏛️ Ministry of Education – Central GFTI",
        "💰 Low fee structure",
    ],
    "Gautam Buddha University": [
        "📍 Greater Noida – Delhi NCR location",
        "🏛️ State University – UP",
        "🎓 NAAC accredited",
    ],
    "Institute of Chemical Technology, Mumbai: Indian Oil Odisha Campus, Bhubaneswar": [
        "🏭 ICT Mumbai collaboration",
        "📍 Located in Bhubaneswar",
        "🔬 Chemical & process engineering focus",
        "🏛️ Govt-funded",
    ],
    "Institute of Infrastructure, Technology, Research and Management": [
        "📍 Located in Ahmedabad, Gujarat",
        "🏗️ Infrastructure & management engineering focus",
        "🏛️ Gujarat Govt. institute",
    ],
    "National Institute of Electronics and Information Technology": [
        "🏛️ Ministry of Electronics & IT (MeitY) institute",
        "💻 Specialised in electronics & IT education",
        "🔬 Industry-linked curriculum",
        "💰 Affordable fee structure",
    ],
    "National Institute of Food Technology Entrepreneurship and Management": [
        "🍱 Unique food technology & entrepreneurship focus",
        "🏛️ Ministry of Food Processing Industries",
        "💼 Strong FMCG & food sector placements",
        "💰 Affordable fee structure",
    ],
    "National Institute of Food Technology, Entrepreneurship and Management (NIFTEM) - Thanjavur": [
        "🍱 Food technology & entrepreneurship focus",
        "🏛️ Ministry of Food Processing Industries",
        "📍 Located in Thanjavur, Tamil Nadu",
    ],
    "Indian Institute of Carpet Technology": [
        "🎨 Unique specialisation in carpet & textile technology",
        "📍 Located in Bhadohi – carpet capital of India",
        "🏛️ Ministry of Textiles institute",
    ],
    "Indian Institute of Handloom Technology(IIHT), Varanasi": [
        "🧵 Specialised in handloom & textile technology",
        "🏛️ Ministry of Textiles institute",
        "📍 Located in Varanasi",
    ],
    "Indian Institute of Handloom Technology, Salem": [
        "🧵 Specialised in handloom & textile technology",
        "🏛️ Ministry of Textiles institute",
        "📍 Located in Salem, Tamil Nadu",
    ],
    "Indian Maritime University - Visakhapatnam Campus": [
        "🚢 India's premier maritime university",
        "🏛️ Ministry of Ports, Shipping & Waterways",
        "💼 Direct maritime industry placements",
        "🌊 Coastal campus in Vizag",
    ],
    "Chhattisgarh Swami Vivekanada Technical University": [
        "📍 Located in Bhilai, Chhattisgarh",
        "🏛️ State Technical University",
        "💰 Low fee structure",
    ],
    "Gati Shakti Vishwavidyalaya": [
        "🚀 India's first multi-modal transport university",
        "🏛️ Ministry of Railways institute",
        "📍 Located in Vadodara, Gujarat",
        "💼 Strong Railways & logistics sector placements",
    ],
    "J.K. Institute of Applied Physics & Technology": [
        "📍 Located in Allahabad (Prayagraj)",
        "🎓 University of Allahabad",
        "🔬 Applied physics & electronics focus",
    ],
    "lndian Institute of Food Processing Technology": [
        "🍱 Premier food processing technology institute",
        "📍 Located in Thanjavur, Tamil Nadu",
        "🏛️ Ministry of Food Processing Industries",
    ],
    "Central institute of Technology Kokrajar": [
        "🏛️ Centrally funded – free tuition for SC/ST",
        "📍 Located in Kokrajhar, Assam",
        "💰 Very affordable fee structure",
    ],
}

