"""
Dedicated database for GFTI Fee Structures (2025-2026 Academic Session).
Contains detailed fee data for top GFTIs and templates for others to be populated.
"""

GFTI_FEE_DATA: dict[str, dict] = {
    # ── Populated / Researched GFTIs ──────────────────────────────────────────

    "sant longowal institute of engineering and technology": {
        "short_name": "SLIET Longowal",
        "program": "B.E. (Bachelor of Engineering)",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Caution Money (Refundable)", "amount": 5000},
            {"item": "Admission Fee", "amount": 5000},
            {"item": "Student Activity Fee", "amount": 5000},
            {"item": "Library Fee", "amount": 5000},
            {"item": "Alumni Fee", "amount": 2000},
        ],
        "semester_fees": [
            {"item": "Tuition Fee", "amount": 20000},
            {"item": "Development Fee", "amount": 5000},
            {"item": "Examination Fee", "amount": 3000},
            {"item": "Other Charges", "amount": 5000},
        ],
        "hostel_fees": [
            {"type": "Multiple Occupancy", "amount_per_sem": 14500},
            {"type": "Single Occupancy", "amount_per_sem": 15500},
        ],
        "additional_notes": [
            "Tuition fee waiver applicable as per Govt of India norms for SC/ST students.",
            "Hostel allocation is subject to availability."
        ],
        "source_url": "http://sliet.ac.in/",
        "source_pdf": ""
    },

    "punjab engineering college": {
        "short_name": "PEC Chandigarh",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Admission Fee", "amount": 5000},
            {"item": "One-Time Student Service Fee", "amount": 6000},
            {"item": "Security Deposit (Refundable)", "amount": 8000},
        ],
        "semester_fees": [
            {"item": "Academic / Tuition Fee", "amount": 88000},
            {"item": "Facilities & Services", "amount": 6000},
        ],
        "hostel_fees": [
            {"type": "Hostel Fee & Mess Establishment", "amount_per_sem": 62600},
            {"type": "Maintenance Charges", "amount_per_sem": 10000},
        ],
        "additional_notes": [
            "Hostel accommodation is generally NOT provided to students admitted under the Chandigarh Quota.",
            "Security deposit is refundable at the end of the course."
        ],
        "source_url": "https://www.pec.ac.in/admissions",
        "source_pdf": ""
    },

    "delhi technological university": {
        "short_name": "DTU Delhi",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Admission / Security", "amount": 10000},
        ],
        "semester_fees": [
            {"item": "Annual Tuition & Fees (Divided per sem)", "amount": 123850},
        ],
        "hostel_fees": [
            {"type": "Hostel Seat Rent & Amenities", "amount_per_sem": 35000},
        ],
        "additional_notes": [
            "DTU charges fees annually, listed above as an estimated semester division for parity.",
            "Total annual fee for 2024-25 is approx ₹2,47,700.",
            "Tuition fee increases slightly every year (e.g., ₹1,63,000 for 25-26)."
        ],
        "source_url": "https://dtu.ac.in/",
        "source_pdf": ""
    },

    "birla institute of technology, mesra": {
        "short_name": "BIT Mesra",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Admission Fee", "amount": 17500},
            {"item": "Caution Money (Refundable)", "amount": 10000},
        ],
        "semester_fees": [
            {"item": "Tuition Fee", "amount": 146500},
            {"item": "Development Fee", "amount": 10000},
            {"item": "Institute Exam Fee", "amount": 7000},
        ],
        "hostel_fees": [
            {"type": "Hostel Seat Rent & Electricity", "amount_per_sem": 20000},
            {"type": "Mess Advance", "amount_per_sem": 18000},
        ],
        "additional_notes": [
            "Tuition fee increases incrementally each semester.",
            "Hostel is mandatory for all students at the Mesra main campus."
        ],
        "source_url": "https://www.bitmesra.ac.in/",
        "source_pdf": ""
    },

    "indian institute of engineering science and technology, shibpur": {
        "short_name": "IIEST Shibpur",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Institute Caution Money", "amount": 3000},
            {"item": "Admission Fee", "amount": 500},
            {"item": "Infrastructure Maintenance", "amount": 2500},
        ],
        "semester_fees": [
            {"item": "Tuition Fee", "amount": 62500},
            {"item": "Examination Fee", "amount": 1000},
            {"item": "Student Activity", "amount": 500},
        ],
        "hostel_fees": [
            {"type": "Seat Rent", "amount_per_sem": 1000},
            {"type": "Mess Advance", "amount_per_sem": 17500},
        ],
        "additional_notes": [
            "100% Tuition Fee waiver for SC/ST/PwD students.",
            "Full/partial waivers available for economically weaker sections (EWS) per Govt. norms."
        ],
        "source_url": "https://www.iiests.ac.in/",
        "source_pdf": ""
    },


    # ── Templates for Remaining GFTIs (TODO) ──────────────────────────────────
    "netaji subhas university of technology": {
        "short_name": "NSUT Delhi",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Admission / Security", "amount": 10000},
        ],
        "semester_fees": [
            {"item": "Annual Fee (Estimated per sem)", "amount": 120000},
        ],
        "hostel_fees": [],
        "additional_notes": [
            "Total annual fee for first-year B.Tech is approx ₹2,40,000.",
            "Hostel and mess fees are charged separately.",
        ],
        "source_url": "https://www.nsut.ac.in/",
        "source_pdf": ""
    },
    "indira gandhi delhi technical university for women": {
        "short_name": "IGDTUW Delhi",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Admission / Security", "amount": 5000},
        ],
        "semester_fees": [
            {"item": "Annual Fee (Estimated per sem)", "amount": 79000},
        ],
        "hostel_fees": [],
        "additional_notes": [
            "Total 4-year fee is approximately ₹7.07 Lakhs.",
            "First year fee is approx ₹1.58 Lakhs to ₹2.65 Lakhs.",
            "100% tuition fee waiver available for high JEE ranks and EWS categories."
        ],
        "source_url": "https://www.igdtuw.ac.in/",
        "source_pdf": ""
    },
    "institute of chemical technology, mumbai": {
        "short_name": "ICT Mumbai",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Caution Money & Deposits", "amount": 2000},
        ],
        "semester_fees": [
            {"item": "Tuition Fee (Per Sem estimated)", "amount": 42500},
        ],
        "hostel_fees": [],
        "additional_notes": [
            "Fee structure varies significantly based on domicile (Maharashtra vs All India) and category.",
            "Hostel and mess are separate and optional."
        ],
        "source_url": "https://ictmumbai.edu.in/",
        "source_pdf": ""
    },
    "shri mata vaishno devi university": {
        "short_name": "SMVDU Katra",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Other University Charges", "amount": 20000},
        ],
        "semester_fees": [
            {"item": "Annual Tuition Fee (Estimated per sem)", "amount": 70620},
        ],
        "hostel_fees": [
            {"type": "Hostel Charges (Annual)", "amount_per_sem": 17000},
        ],
        "additional_notes": [
            "Annual tuition fee is Rs. 1,41,240.",
            "Hostel charges are Rs. 17,000 annually. Mess fees are separate."
        ],
        "source_url": "https://www.smvdu.ac.in/",
        "source_pdf": ""
    },
    "school of planning & architecture": {
        "short_name": "SPA Delhi",
        "program": "B.Arch / B.Plan",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Enrolment Fee", "amount": 5000},
            {"item": "Security Deposit (Refundable)", "amount": 20000},
        ],
        "semester_fees": [
            {"item": "Tuition Fee", "amount": 42500},
            {"item": "Registration Fee", "amount": 1000},
            {"item": "Academic Support & Student Activities", "amount": 8000},
        ],
        "hostel_fees": [
            {"type": "Hostel Rent & Electricity (Annual)", "amount_per_sem": 22000},
        ],
        "additional_notes": [
            "Tuition fee varies by semester (Rs 40k for odd sems, Rs 45k for even sems).",
            "SC/ST category tuition fee is 50% reduced (Rs 20k/22.5k)."
        ],
        "source_url": "https://www.spa.ac.in/",
        "source_pdf": ""
    },
    "assam university": {
        "short_name": "Assam Univ (TSSOT)",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Admission & Placement Fees", "amount": 5000},
        ],
        "semester_fees": [
            {"item": "Tuition Fee & Development Funds (Estimated per sem)", "amount": 29460},
        ],
        "hostel_fees": [
            {"type": "Hostel Fee (Annual)", "amount_per_sem": 25000},
        ],
        "additional_notes": [
            "Total payable amount at the time of admission is approximately ₹58,920.",
            "Registration and examination fees are charged separately."
        ],
        "source_url": "https://www.aus.ac.in/",
        "source_pdf": ""
    },
    "pondicherry engineering college": {
        "short_name": "PTU Puducherry",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Admission & Registration (Estimated)", "amount": 5000},
        ],
        "semester_fees": [
            {"item": "Annual Fee for JoSAA/CSAB (Self-Support)", "amount": 162401},
        ],
        "hostel_fees": [],
        "additional_notes": [
            "Fees depend heavily on category (CENTAC vs JoSAA).",
            "JoSAA/CSAB self-support seats generally range from ₹87k to ₹1.6L annually."
        ],
        "source_url": "https://ptuniv.edu.in/",
        "source_pdf": ""
    },
    "pandit deendayal energy university": {
        "short_name": "PDEU Gujarat",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Library & Security Deposit", "amount": 13000},
        ],
        "semester_fees": [
            {"item": "Tuition Fee", "amount": 124500},
            {"item": "University Enrollment Fee", "amount": 3000},
        ],
        "hostel_fees": [
            {"type": "Hostel & Mess (Annual approx)", "amount_per_sem": 148000},
        ],
        "additional_notes": [
            "Merit-based tuition fee waivers available for high JEE Main ranks.",
            "Hostel and mess combined cost is approx ₹1.48 Lakhs annually."
        ],
        "source_url": "https://pdeu.ac.in/",
        "source_pdf": ""
    },
    "manipal institute of technology": {
        "short_name": "MIT Manipal",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Caution Deposit (Refundable)", "amount": 10000},
        ],
        "semester_fees": [
            {"item": "Tuition & Dev Fee (Estimated per sem)", "amount": 200000},
        ],
        "hostel_fees": [
            {"type": "Hostel & Mess (Depends on AC/Non-AC)", "amount_per_sem": 75000},
        ],
        "additional_notes": [
            "Total 4-year tuition fee ranges from ₹14L to ₹20L depending on the branch.",
            "Hostel costs vary heavily based on room type selection."
        ],
        "source_url": "https://manipal.edu/",
        "source_pdf": ""
    },
    "jawaharlal nehru university": {
        "short_name": "JNU Delhi",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Admission & Alumni Fee", "amount": 2000},
            {"item": "Security Deposit (Refundable)", "amount": 5000},
        ],
        "semester_fees": [
            {"item": "Tuition Fee (Income > 5 Lakhs)", "amount": 62500},
            {"item": "Student Activity & Dev Fund", "amount": 3500},
            {"item": "Exam & Registration Fee", "amount": 2000},
        ],
        "hostel_fees": [],
        "additional_notes": [
            "Tuition fee is ₹0 for SC/ST/PwD and Income < ₹1 Lac.",
            "Tuition fee is ₹20,833 for Income ₹1–5 Lacs.",
            "Hostel charges are separate and deposited at allotment."
        ],
        "source_url": "https://www.jnu.ac.in/",
        "source_pdf": ""
    },
    "university of hyderabad": {
        "short_name": "UoH Hyderabad",
        "program": "Integrated B.Tech + M.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Caution Deposits", "amount": 2570},
            {"item": "Students Union / Aid Fund", "amount": 840},
        ],
        "semester_fees": [
            {"item": "Tuition Fee", "amount": 21705},
            {"item": "Other Fee (Lab/Academic)", "amount": 10475},
        ],
        "hostel_fees": [],
        "additional_notes": [
            "Total fee for the first semester is approximately Rs. 37,980.",
            "UoH offers 5-year integrated programs, not a standalone 4-year B.Tech."
        ],
        "source_url": "https://uohyd.ac.in/",
        "source_pdf": ""
    },
    "tezpur university": {
        "short_name": "Tezpur Univ",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Caution Deposit & Reg", "amount": 5000},
        ],
        "semester_fees": [
            {"item": "Tuition & Misc Fees (Estimated)", "amount": 30000},
        ],
        "hostel_fees": [
            {"type": "Hostel Seat Rent (Annual approx)", "amount_per_sem": 15000},
        ],
        "additional_notes": [
            "Food Engineering B.Tech students pay an additional consumables fee.",
            "SC/ST students may be exempt from hostel seat rent."
        ],
        "source_url": "https://tezu.ernet.in/",
        "source_pdf": ""
    },
    "national institute of electronics and information technology": {
        "short_name": "NIELIT",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Caution Money (Estimated)", "amount": 5000},
        ],
        "semester_fees": [
            {"item": "Tuition Fee (Approx per sem)", "amount": 50000},
        ],
        "hostel_fees": [],
        "additional_notes": [
            "Fee structure varies heavily by specific center (Aurangabad, Patna, Ropar, etc.).",
            "SC/ST students generally receive tuition fee waivers per GoI norms."
        ],
        "source_url": "https://nielit.gov.in/",
        "source_pdf": ""
    },
    "national institute of food technology": {
        "short_name": "NIFTEM",
        "program": "B.Tech",
        "academic_year": "2025-26",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Security Deposit (Refundable)", "amount": 15000},
            {"item": "Alumni Membership & ID", "amount": 11650},
        ],
        "semester_fees": [
            {"item": "Tuition Fee", "amount": 51650},
            {"item": "Other Charges (Enrolment, Lab, Exam)", "amount": 70450},
        ],
        "hostel_fees": [
            {"type": "Hostel Room Rent (Twin Sharing)", "amount_per_sem": 6000},
            {"type": "Mess Charges (Advance)", "amount_per_sem": 25800},
        ],
        "additional_notes": [
            "SC/ST/PWD pay ₹0 tuition fee.",
            "2/3rd tuition waiver for family income between 1-5 lakhs."
        ],
        "source_url": "https://niftem.ac.in/",
        "source_pdf": ""
    },
    "gati shakti vishwavidyalaya": {
        "short_name": "GSV Vadodara",
        "program": "B.Tech",
        "academic_year": "2024-25",
        "currency": "₹",
        "one_time_fees": [
            {"item": "Caution Deposit (Refundable)", "amount": 10000},
            {"item": "Admission Fee", "amount": 2000},
        ],
        "semester_fees": [
            {"item": "Annual Tuition Fee", "amount": 153100},
            {"item": "Annual Registration & Med Insurance", "amount": 6000},
            {"item": "Campus Facilities Charge (Annual)", "amount": 5000},
        ],
        "hostel_fees": [
            {"type": "Annual Hostel Fee (Double Sharing)", "amount_per_sem": 43600},
            {"type": "Annual Mess & Electricity", "amount_per_sem": 46000},
        ],
        "additional_notes": [
            "Fees are mostly structured annually instead of per-semester.",
            "Hostel and mess are mandatory."
        ],
        "source_url": "https://gsv.ac.in/",
        "source_pdf": ""
    },
}
