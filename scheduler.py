import csv
import random
import math


class ServiceSite:
    def __init__(self, site_name: str, availability: list, is_outdoors: bool, difficulty: int, minors_allowed=True):
        self.site_name = site_name
        self.availability = availability
        self.is_outdoors = is_outdoors
        self.minors_allowed = minors_allowed

        # 1=easy, 2=moderate, 3=hard
        self.difficulty = difficulty

        self.schedule = [0] * 5

    def not_available(self, slot):
        if self.schedule[slot] >= self.availability[slot]:
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
NUM_SLOTS = 5

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
    if site.is_outdoors and slot != NUM_SLOTS - 1:  # Check that it's not the last slot
        same_day_diff_time = slot ^ 1  # XOR to toggle between 0<->1, 2<->3
        if crew.schedule[same_day_diff_time] is not None:
            scheduled_site_idx = crew.schedule[same_day_diff_time]
            if SERVICE_SITES[scheduled_site_idx].is_outdoors:
                return False
            
    # Check if the crew has already volunteered at this site
    if site_idx in crew.schedule:
        return False

    return True


def is_valid_from_grid(schedule, crew_idx, slot, crews, sites):
    crew = crews[crew_idx]
    site_idx = schedule[crew_idx][slot]
    site = sites[site_idx]

    # Check crew's blacklist
    if site_idx in crew.blacklist:
        return False
    
    # Check if the crew has too many outdoor sites
    if (site_idx != 5) and (site.is_outdoors) and sites[site_idx ^ 1].is_outdoors:
        return False
    
    # Check if the site is overbooked
    for site_slot in range(NUM_SLOTS):
        if site.schedule[site_slot] > site.availability[site_slot]:
            return False
        
    # Check whether the crew has already volunteered at the site
    if schedule[crew_idx].count(site_idx) > 1:
        return False

    return True


def get_score_from_grid(schedule, crew_idx, slot, crews, sites):
    score = 0
    crew = crews[crew_idx]
    site_idx = schedule[crew_idx][slot]
    site = sites[site_idx]

    # Check crew's blacklist
    if site_idx in crew.blacklist:
        score += 5
    
    # Check if the crew has too many outdoor sites
    if site.is_outdoors:
        if (slot == 0 and sites[schedule[crew_idx][1]].is_outdoors
            or slot == 1 and sites[schedule[crew_idx][0]].is_outdoors
            or slot == 2 and sites[schedule[crew_idx][3]].is_outdoors
            or slot == 3 and sites[schedule[crew_idx][2]].is_outdoors):
        
            score += 5
    
    # Check if the site is overbooked
    for site_slot in range(NUM_SLOTS):
        if site.schedule[site_slot] > site.availability[site_slot]:
            score += 1000
        
    # Check whether the crew has already volunteered at the site
    if schedule[crew_idx].count(site_idx) > 1:
        score += 1000

    return score


def get_cost(schedule, crews, sites):
    cost = 0
    for crew_idx in range(len(crews)):
        for slot in range(NUM_SLOTS):
            cost += get_score_from_grid(schedule, crew_idx, slot, crews, sites)
    return cost


def generate_all_neighbors(schedule, crews, sites):
    """Generate all neighbors by changing one timeslot at a time to each other possible site."""
    neighbors = []
    n_crews = len(crews)
    n_sites = len(sites)

    for crew_idx in range(n_crews):
        for slot in range(NUM_SLOTS):
            current_site_idx = schedule[crew_idx][slot]
            for new_site_idx in range(n_sites):
                if new_site_idx != current_site_idx:
                    # Create a new neighbor by modifying the current site
                    neighbor = [crew_schedule[:] for crew_schedule in schedule]
                    neighbor[crew_idx][slot] = new_site_idx
                    neighbors.append(neighbor)

    return neighbors


def generate_random_neighbors(schedule, crews, sites, num_neighbors=100, num_changes=100):
    """Generate a number of random neighbors by making a small number of changes."""
    neighbors = []
    n_crews = len(crews)
    n_sites = len(sites)  # Number of sites

    for _ in range(num_neighbors):
        neighbor = [crew_schedule[:] for crew_schedule in schedule]
        
        # Make random changes
        for _ in range(num_changes):
            crew_idx = random.randint(0, n_crews - 1)
            slot = random.randint(0, NUM_SLOTS - 1)
            new_site_idx = random.randint(0, n_sites - 1)
            
            # Apply the change if it's different
            if neighbor[crew_idx][slot] != new_site_idx:
                old_site_idx = neighbor[crew_idx][slot]
                
                # Update the schedule and site bookings
                neighbor[crew_idx][slot] = new_site_idx
                
                # Optionally, update site bookings if needed
                # (You may need to handle this according to your scheduling logic)
                
                # Optionally revert old site booking if needed
                # (You may need to handle this according to your scheduling logic)

        neighbors.append(neighbor)

    return neighbors


def first_fit(crews, sites):
    slot = 0
    site_idx = 0
    crew_idx = 0

    while (crew_idx < len(crews)) and (slot < NUM_SLOTS):  # stop when we've reached last crew's slot
        site = sites[site_idx]
        crew = crews[crew_idx]
        # print(f"(slot: {slot}, site_idx: {site_idx}, crew_idx: {crew_idx})")
        if is_valid_assignment(crew_idx, site_idx, slot, crews, sites):
            site.schedule[slot] += 1
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
            site.schedule[slot] += 1
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
                sites[site_idx].schedule[slot] -= 1
                crews[crew_idx].schedule[slot] = None
            site_idx += 1

    return False  # If the stack is empty and we didn't return True, scheduling failed


def simple_hill_climbing(crews, sites):
    initial_solution = []

    for crew in range(len(crews)):
        crew_slots = []
        for slot in range(NUM_SLOTS):
            site_idx = random.randint(0, len(sites) - 1)
            crew_slots.append(site_idx)
            sites[site_idx].schedule[slot] += 1
            crews[crew].schedule[slot] = site_idx

        initial_solution.append(crew_slots)
    
    # Step 2: Hill Climbing Loop
    current_solution = initial_solution
    current_cost = get_cost(current_solution, crews, sites)

    while current_cost > 0:
        # Generate all neighboring solutions
        neighbors = generate_all_neighbors(current_solution, crews, sites)
        neighbor_costs = [get_cost(neighbor, crews, sites) for neighbor in neighbors]

        # Find the best neighbor (lowest cost)
        min_cost = min(neighbor_costs)
        print(min_cost)
        min_cost_index = neighbor_costs.index(min_cost)

        # If the best neighbor is better than the current solution, move to it
        if min_cost < current_cost:
            current_solution = neighbors[min_cost_index]
            current_cost = min_cost
        else:
            # No better neighbor found, stop searching
            break

    return current_solution


def list_of_lists_to_tuple_of_tuples(list_of_lists):
    return tuple(tuple(inner_list) for inner_list in list_of_lists)


def greedy_initial_solution(crews, sites):
    """Generate an initial solution using a greedy approach."""
    schedule = [[] for _ in range(len(crews))]

    for crew_idx, crew in enumerate(crews):
        for slot in range(NUM_SLOTS):
            for site_idx, site in enumerate(sites):
                if not site.not_available(slot) and site_idx not in crew.blacklist:
                    schedule[crew_idx].append(site_idx)
                    site.schedule[slot] += 1
                    break

    return schedule


def apply_schedule(solution, crews, sites):
    # Reset site and crew schedules
    for site in sites:
        site.schedule = [0] * NUM_SLOTS

    for crew in crews:
        crew.schedule = [None] * NUM_SLOTS

    # Apply the solution
    for crew_idx, crew_slots in enumerate(solution):
        crew = crews[crew_idx]
        for slot, site_idx in enumerate(crew_slots):
            site = sites[site_idx]
            site.schedule[slot] += 1
            crew.schedule[slot] = site_idx


def tabu_search(crews, sites, max_iterations=10000, plateau_limit=50, restart_interval=1000):
    initial_solution = greedy_initial_solution(crews, sites)

    tabu_set = set()
    current_solution = initial_solution
    current_cost = get_cost(current_solution, crews, sites)
    plateau_counter = 0
    best_solution = current_solution
    best_cost = current_cost

    iteration = 0
    while iteration < max_iterations and best_cost > 0:
        neighbors = generate_random_neighbors(current_solution, crews, sites)

        best_neighbor = None
        best_neighbor_cost = float('inf')
        all_neighbor_costs = []

        for neighbor in neighbors:
            neighbor_tuple = list_of_lists_to_tuple_of_tuples(neighbor)
            if neighbor_tuple not in tabu_set:
                neighbor_cost = get_cost(neighbor, crews, sites)
                all_neighbor_costs.append(neighbor_cost)
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost

        print(f"Iteration {iteration}: Best Cost in Neighbors = {best_neighbor_cost}")

        if best_neighbor is not None and best_neighbor_cost < current_cost:
            current_solution = best_neighbor
            apply_schedule(current_solution, crews, sites)
            current_cost = best_neighbor_cost
            tabu_set.add(list_of_lists_to_tuple_of_tuples(current_solution))
            plateau_counter = 0

            # Update best solution found
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
        else:
            plateau_counter += 1

            if plateau_counter > plateau_limit:
                print("PLATEAU")
                for neighbor in neighbors:
                    neighbor_tuple = list_of_lists_to_tuple_of_tuples(neighbor)
                    if neighbor_tuple not in tabu_set:
                        neighbor_cost = get_cost(neighbor, crews, sites)
                        if neighbor_cost > current_cost:
                            current_solution = neighbor
                            current_cost = neighbor_cost
                            tabu_set.add(neighbor_tuple)
                            plateau_counter = 0
                            break

        iteration += 1

    for sol in best_solution:
        print(sol)

    return best_solution


def simulated_annealing(crews, sites, max_iterations=100000, initial_temperature=10000, cooling_rate=0.995):
    def generate_neighbor(schedule):
        """Generate a neighbor by randomly changing a small subset of the schedule."""
        neighbor = [crew_schedule[:] for crew_schedule in schedule]
        num_changes = random.randint(1, 3)  # Change a random number of slots (e.g., 1 to 3)
        for _ in range(num_changes):
            crew_idx = random.randint(0, len(crews) - 1)
            slot_idx = random.randint(0, NUM_SLOTS - 1)
            new_site_idx = random.randint(0, len(sites) - 1)
            neighbor[crew_idx][slot_idx] = new_site_idx
        return neighbor

    def acceptance_probability(current_cost, neighbor_cost, temperature):
        """Calculate the acceptance probability based on cost difference and temperature."""
        if neighbor_cost < current_cost:
            return 1.0
        return math.exp((current_cost - neighbor_cost) / temperature)

    def update_schedules(schedule, crews, sites):
        """Update crew and site schedules based on the given solution."""
        def reset_schedules(crews, sites):
            """Reset crew and site schedules to initial state."""
            for site in sites:
                site.schedule = [0] * NUM_SLOTS
            for crew in crews:
                crew.schedule = [None] * NUM_SLOTS
        
        reset_schedules(crews, sites)
        for crew_idx, crew_schedule in enumerate(schedule):
            for slot, site_idx in enumerate(crew_schedule):
                site = sites[site_idx]
                site.schedule[slot] += 1
                crews[crew_idx].schedule[slot] = site_idx

    # Initialize the solution
    current_solution = greedy_initial_solution(crews, sites)
    update_schedules(current_solution, crews, sites)
    current_cost = get_cost(current_solution, crews, sites)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temperature

    iteration = 0
    while iteration < max_iterations and best_cost > 0:
        print(f"iteration: {iteration}; cost: {best_cost}")
        neighbor_solution = generate_neighbor(current_solution)
        update_schedules(neighbor_solution, crews, sites)
        neighbor_cost = get_cost(neighbor_solution, crews, sites)

        # Decide whether to accept the neighbor
        if acceptance_probability(current_cost, neighbor_cost, temperature) > random.random():
            current_solution = neighbor_solution
            update_schedules(current_solution, crews, sites)
            current_cost = neighbor_cost

            # Update best solution found
            if current_cost < best_cost:
                best_solution = current_solution
                update_schedules(best_solution, crews, sites)
                best_cost = current_cost

        # Cool down the temperature
        temperature *= cooling_rate
        iteration += 1

    # Print final results
    for sol in best_solution:
        print(sol)

    update_schedules(best_solution, crews, sites)
    return best_solution


def list_problems(schedule, crews, sites):
    problems = []
    
    for crew_idx in range(len(crews)):
        crew = crews[crew_idx]
        for slot in range(NUM_SLOTS):
            site_idx = schedule[crew_idx][slot]
            site = sites[site_idx]
            
            # Check crew's blacklist
            if site_idx in crew.blacklist:
                problems.append(f"Crew {crew_idx + 1} at slot {slot}: {sites[site_idx].site_name} is in blacklist.")
            
            # Check if the crew has too many outdoor sites
            if (site_idx != 5) and (site.is_outdoors) and sites[site_idx ^ 1].is_outdoors:
                problems.append(f"Crew {crew_idx + 1} at slot {slot}: {sites[site_idx].site_name} and its pair are both outdoors.")
            
            # Check if the site is overbooked
            for site_slot in range(NUM_SLOTS):
                if site.schedule[site_slot] > site.availability[site_slot]:
                    problems.append(f"{sites[site_idx].site_name} at slot {site_slot}: Overbooked.")
                    print(site.site_name)
                    print(site.schedule)
                    print(site.availability)
            
            # Check whether the crew has already volunteered at the site
            if schedule[crew_idx].count(site_idx) > 1:
                problems.append(f"Crew {crew_idx + 1}: {sites[site_idx].site_name} is visited more than once.")
                
    return problems


if __name__ == "__main__":
    random.seed(43)

    create_blacklists(CREWS, SERVICE_SITES)
    crews = sort_crews_by_restrictions(CREWS)
    sites = sort_sites_by_availability(SERVICE_SITES)

    headers = ["Crew #", "CLs", "Mon AM", "Mon PM", "Tue AM", "Tue PM", "Wed AM"]

    with open('schedule_output.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header row
        csv_writer.writerow(headers)

        # Run Simulated Annealing
        schedule = simulated_annealing(crews, sites)

        problems = list_problems(schedule, crews, sites)
        for problem in problems:
            print(problem)
        # Write the data rows
        for crew_idx in range(len(schedule)):
            crew = crews[crew_idx]
            # Create a row with the crew leader's name followed by their schedule
            row = [crew_idx + 1, crew.crew_leaders]
            # Extend the row with schedule entries corresponding to each header
            for time in range(NUM_SLOTS):
                site_idx = schedule[crew_idx][time]
                row.append(sites[site_idx].site_name if site_idx is not None else "Unassigned")
            csv_writer.writerow(row)
        print("Scheduling successful! Results written to 'schedule_output.csv'.")