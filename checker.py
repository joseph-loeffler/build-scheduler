from collections import defaultdict

data = """	Monday, August 21st from 10:30am to 12:30pm (+5)	Monday, August 21st from 2:00pm to 4:00pm (+3)	Tuesday, August 22nd from 10:30am to 12:30pm	Tuesday, August 22nd from 2:00pm to 4:00pm	Wednesday, August 23rd from 10:30am to 12:30pm
Joe & Pallavi	Durham Parks & Recreation	Reality Ministries (indoor)	TROSA	Iglesia Emanuel	Scrap Exchange
Adama & Kat	Durham Parks & Recreation	TROSA	Reality Ministries (outdoor)	Scrap Exchange	Keep Durham Beautiful
Shivani & Oli	Durham Central Park	Forest View Elementary School	Scrap Exchange	Kramden Institute	Keep Durham Beautiful
John & Laurin	TROSA	Duke Lemur Center	Duke Campus Farm	Forest View Elementary School	Diaper Bank of NC
Sofia & Preston	Children First	Reality Ministries (indoor)	Diaper Bank of NC	TROSA	Durham Central Park
Naomi & Aiden	Iglesia Emanuel	Reality Ministries (indoor)	TROSA	Forest View Elementary School	Diaper Bank of NC
Andy & Morgan	Children First	TROSA	Diaper Bank of NC	Durham Success Summit	W.G. Pearson Center
Rachel & Nathan	Durham Central Park	TROSA	Children First	Durham Success Summit	Keep Durham Beautiful
Jess & Eleanor	TROSA	Reality Ministries (indoor)	Scrap Exchange	Special Olympics	Duke Lemur Center
Ian & Lily	Duke Lemur Center	TROSA	Duke Campus Farm	Forest View Elementary School	Diaper Bank of NC
Ben & Olivia	Housing for New Hope	Forest View Elementary School	TROSA	Bull City Woodshop	Durham Central Park
Enoch & Molly	Durham Parks & Recreation	Bull City Woodshop	SEEDS	TROSA	Iglesia Emanuel
Ale & Dhruv	TROSA	Museum of Life and Science (indoor)	SEEDS	Forest View Elementary School	Hub Farm
Tiki & Ellen	Durham Parks & Recreation	Forest View Elementary School	Hub Farm	TROSA	Scrap Exchange
Ghina & Robbie	TROSA	Reality Ministries (indoor)	Diaper Bank of NC	Museum of Life and Science (outdoor)	W.G. Pearson Center
Julia & Isabel	Durham Parks & Recreation	Iglesia Emanuel	TROSA	Scrap Exchange	Keep Durham Beautiful"""

# Process the tab-separated data
lines = data.strip().split('\n')
schedules = {}

for line in lines[1:]:  # Skip the header row
    parts = line.split('\t')
    group = parts[0].strip()
    services = [service.strip() for service in parts[1:]]
    schedules[group] = services

# Track the unique groups each group volunteers with
volunteer_tracker = defaultdict(set)

# Compare the service sites by timeslot
for group, services in schedules.items():
    for i in range(len(services)):  # Dynamically determine the number of timeslots
        for other_group, other_services in schedules.items():
            if group != other_group and services[i] and services[i] == other_services[i]:
                volunteer_tracker[group].add(other_group)

# Print the crew name and the number of other unique crews they volunteered with
for group, others in volunteer_tracker.items():
    print(f"{group}: {len(others)} other unique crews")