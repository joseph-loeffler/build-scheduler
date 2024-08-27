import csv


class ServiceSite:
    def __init__(self, site_name: str, availability: list, is_outdoors: bool, difficulty: int, minors_allowed=True):
        self.site_name = site_name
        self.availability = availability
        self.is_outdoors = is_outdoors
        self.minors_allowed = minors_allowed

        # 1=easy, 2=moderate, 3=hard
        self.difficulty = difficulty

        self.current_schedule = [0] * 5

    def not_available(self, slot):
        if self.current_schedule[slot] >= self.availability[slot]:
            return True
        else:
            return False


class Crew:
    def __init__(self, crew_leaders: str, has_minor=False, max_difficulty=3, blacklist=None):
        self.crew_leaders = crew_leaders
        self.has_minor = has_minor
        self.max_difficulty = max_difficulty
        self.blacklist = set(blacklist) if blacklist else set([])
        self.schedule = [None] * 5
        self.other_crews = set([])

    def not_available(self, slot):
        if self.schedule[slot] is None:
            return False
        else:
            return True


CREWS = [
    Crew(crew_leaders="Joe & Pallavi", has_minor=True, blacklist=["Museum of Life and Science (outdoor)","Special Olympics"]),  # 1
    Crew(crew_leaders="Adama & Kat", has_minor=True),  # 2
    Crew(crew_leaders="Shivani & Oli", has_minor=True),  # 3
    Crew(crew_leaders="John & Laurin", has_minor=True),  # 4
    Crew(crew_leaders="Sofia & Preston"),  # 5
    Crew(crew_leaders="Naomi & Aiden"),  # 6
    Crew(crew_leaders="Andy & Morgan", has_minor=True, max_difficulty=2),  # 7
    Crew(crew_leaders="Rachel & Nathan"),  # 8
    Crew(crew_leaders="Jess & Eleanor"),  # 9
    Crew(crew_leaders="Ian & Lily"),  # 10
    Crew(crew_leaders="Ben & Olivia"),  # 11
    Crew(crew_leaders="Enoch & Molly"),  # 12
    Crew(crew_leaders="Ale & Dhruv"),  # 13
    Crew(crew_leaders="Tiki & Ellen"),  # 14
    Crew(crew_leaders="Ghina & Robbie", has_minor=True),  # 15
    Crew(crew_leaders="Julia & Isabel", has_minor=True, max_difficulty=2)  # 16
]
SERVICE_SITES = [
    ServiceSite("Diaper Bank of NC", availability=[0,0,3,0,3], is_outdoors=False,  difficulty=1),  # 0
    ServiceSite("Kramden Institute", availability=[0,0,0,1,0], is_outdoors=False,  difficulty=1),  # 1
    ServiceSite("Reality Ministries (indoor)", availability=[0,5,0,0,0], is_outdoors=False,  difficulty=1),  # 2
    ServiceSite("Scrap Exchange", availability=[0,0,2,2,2], is_outdoors=False,  difficulty=1),  # 3
    ServiceSite("TROSA", availability=[4,4,4,3,0], is_outdoors=False,  difficulty=1),  # 4
    ServiceSite("Iglesia Emanuel", availability=[1,1,0,1,1], is_outdoors=False,  difficulty=2),  # 5
    ServiceSite("W.G. Pearson Center", availability=[0,0,0,0,2], is_outdoors=False,  difficulty=2),  # 6
    ServiceSite("Bull City Woodshop", availability=[0,1,0,1,0], is_outdoors=False,  difficulty=2),  # 7
    ServiceSite("Museum of Life and Science (indoor)", availability=[0,1,0,0,0], is_outdoors=False,  difficulty=2),  # 8
    ServiceSite("Durham Success Summit", availability=[0,0,0,2,0], is_outdoors=False,  difficulty=2),  # 9
    ServiceSite("Forest View Elementary School", availability=[0,3,0,4,0], is_outdoors=False,  difficulty=2),  # 10
    ServiceSite("Special Olympics", availability=[0,0,0,1,0], is_outdoors=False, minors_allowed=False, difficulty=1),  # 11
    ServiceSite("Children First", availability=[2,0,1,0,0], is_outdoors=True,  difficulty=2),  # 12
    ServiceSite("Duke Campus Farm", availability=[0,0,2,0,0], is_outdoors=True,  difficulty=3),  # 13
    ServiceSite("Duke Lemur Center", availability=[1,1,0,0,1], is_outdoors=True,  difficulty=3),  # 14
    ServiceSite("Durham Central Park", availability=[2,0,0,0,2], is_outdoors=True,  difficulty=3),  # 15
    ServiceSite("Keep Durham Beautiful", availability=[0,0,0,0,4], is_outdoors=True,  difficulty=3),  # 16
    ServiceSite("SEEDS", availability=[0,0,2,0,0], is_outdoors=True, minors_allowed=False, difficulty=3),  # 17
    ServiceSite("Housing for New Hope", availability=[1,0,0,0,0], is_outdoors=True,  difficulty=2),  # 18
    ServiceSite("Durham Parks & Recreation", availability=[5,0,0,0,0], is_outdoors=True,  difficulty=2),  # 19
    ServiceSite("Museum of Life and Science (outdoor)", availability=[0,0,0,1,0], is_outdoors=True,  difficulty=3),  # 20
    ServiceSite("Hub Farm", availability=[0,0,1,0,1], is_outdoors=True, minors_allowed=False, difficulty=3),  # 21
    ServiceSite("Reality Ministries (outdoor)", availability=[0,0,1,0,0], is_outdoors=True,  difficulty=1)  # 22
]
LAST_SLOT = 4

def create_blacklists(crews, sites):
    for crew_idx in range(len(crews)):
        crew = crews[crew_idx]
        for site_idx in range(len(sites)):
            site = sites[site_idx]
            if crew.has_minor and not site.minors_allowed:
                crew.blacklist.add(site_idx)
            elif site.difficulty > crew.max_difficulty:
                crew.blacklist.add(site_idx)


def sort_crews_by_restrictions(crews):
    def len_blacklist(crew):
        return len(crew.blacklist)
    
    # Sort crews by their restriction score in descending order
    return sorted(crews, key=len_blacklist, reverse=True)


def sort_sites_by_availability(sites):
    def num_of_available_spots(site):
        return sum(site.availability)
    return sorted(sites, key=num_of_available_spots)


def is_valid_assignment(crew_idx: int, site_idx: int, slot: int, crews: list[Crew], sites: list[ServiceSite]):
    crew = crews[crew_idx]
    site = sites[site_idx]

    # Check crew's blacklist
    if site_idx in crew.blacklist:
        return False

    # Check if the crew is already scheduled or the site is full.
    if crew.not_available(slot) or site.not_available(slot):
        return False

    # Check if crew already has too many outdoor sites
    if site.is_outdoors and slot != LAST_SLOT:
        same_day_diff_time = slot ^ 1  # XOR to toggle between 0<->1, 2<->3
        if crew.schedule[same_day_diff_time] is not None:
            scheduled_site_idx = crew.schedule[same_day_diff_time]
            if SERVICE_SITES[scheduled_site_idx].is_outdoors:
                return False
            
    # Check if the crew has already volunteered at this site
    if site_idx in crew.schedule:
        return False

    return True


def first_fit(crews, sites):
    slot = 0
    site_idx = 0
    crew_idx = 0

    while (crew_idx < len(crews)) and (slot < LAST_SLOT + 1):  # stop when we've reached last crew's slot
        site = sites[site_idx]
        crew = crews[crew_idx]
        # print(f"(slot: {slot}, site_idx: {site_idx}, crew_idx: {crew_idx})")
        if is_valid_assignment(crew_idx, site_idx, slot, crews, sites):
            site.current_schedule[slot] += 1
            crew.schedule[slot] = site_idx

            if slot < 4:
                slot += 1
                site_idx = 0
            elif crew_idx < len(crews) - 1:
                crew_idx += 1
                slot = 0
                site_idx = 0
        elif site_idx != len(sites) - 1:  # if it's not valid and we can increment site_idx by 1
            site_idx += 1
        else:  # not valid so we SKIP
            crew.schedule[slot] = None

            if slot < 4:
                slot += 1
                site_idx = 0
            else:
                crew_idx += 1
                slot = 0
                site_idx = 0

    return True  # Return true regardless of whether all constraints were met


def brute_force(crews, sites):
    stack = [(None, None, None)]  # (slot, site_idx, crew_idx)

    slot = 0
    site_idx = 0
    crew_idx = 0

    while stack:  # if stack is empty, scheduling failed
        site = sites[site_idx]
        crew = crews[crew_idx]
        # print(f"(slot: {slot}, site_idx: {site_idx}, crew_idx: {crew_idx})")
        if is_valid_assignment(crew_idx, site_idx, slot, crews, sites):

            stack.append((slot, site_idx, crew_idx))
            site.current_schedule[slot] += 1
            crew.schedule[slot] = site_idx

            if slot < 4:
                slot += 1
                site_idx = 0
            elif crew_idx < len(crews) - 1:
                crew_idx += 1
                slot = 0
                site_idx = 0
            else:
                return True  # If the last slot of the last crew is valid, then return True, scheduling succeeded
        
        elif site_idx != len(sites) - 1:  # if it's not valid and we can increment site_idx by 1
            site_idx += 1
        else:  # not valid and we have to BACKTRACK
            # print("BACKTRACK")
            while site_idx == len(sites) - 1:
                slot, site_idx, crew_idx = stack.pop()
                sites[site_idx].current_schedule[slot] -= 1
                crews[crew_idx].schedule[slot] = None
            site_idx += 1

    return False  # If the stack is empty and we didn't return True, scheduling failed


if __name__ == "__main__":
    create_blacklists(CREWS, SERVICE_SITES)
    crews = sort_crews_by_restrictions(CREWS)
    sites = sort_sites_by_availability(SERVICE_SITES)

    headers = ["Crew #", "CLs", "Mon AM", "Mon PM", "Tue AM", "Tue PM", "Wed AM"]
    
    with open('schedule_output.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header row
        csv_writer.writerow(headers)
        
        # if first_fit(crews, sites):
        if brute_force(crews, sites):
            # Write the data rows
            crew_num = 1
            for crew in crews:
                # Create a row with the crew leader's name followed by their schedule
                row = [crew_num, crew.crew_leaders]
                # Extend the row with schedule entries corresponding to each header
                for time in range(LAST_SLOT + 1):
                    site_idx = crew.schedule[time] if crew.schedule[time] is not None else None
                    row.append(sites[site_idx].site_name if site_idx is not None else "Unassigned")
                csv_writer.writerow(row)
            print("Scheduling successful! Results written to 'schedule_output.csv'.")
        else:
            print("Failed to find a valid schedule.")